from transformers import AutoModel
import torch.nn as nn

class DINOv2KneeTransformer(nn.Module):
    def __init__(self, dinov2_path: Path, num_labels: int = 12,
                 n_slices: int = 9, embed_dim: int = 384):
        super().__init__()

        # ── DINOv2 backbone ───────────────────────────────────────────────────
        self.backbone  = AutoModel.from_pretrained(str(dinov2_path))
        self.embed_dim = embed_dim

        # ── Plane embedding ───────────────────────────────────────────────────
        self.plane_embedding = nn.Embedding(3, embed_dim)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = x.shape

        x        = x.view(B * S, C, H, W)
        outputs  = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]
        features = features.view(B, S, self.embed_dim)

        plane_ids = torch.arange(3, device=x.device).repeat_interleave(N_SLICES)
        plane_emb = self.plane_embedding(plane_ids)
        features  = features + plane_emb.unsqueeze(0)

        features = self.transformer(features)
        pooled   = features.mean(dim=1)

        return self.classifier(pooled)


# ── Instantiate ───────────────────────────────────────────────────────────────
model = DINOv2KneeTransformer(DINOV2_PATH, num_labels=NUM_LABELS).to(device)
model = torch.compile(model)
# ── Freeze backbone initially ─────────────────────────────────────────────────
for param in model.backbone.parameters():
    param.requires_grad = False

print("Model ready ✓  Backbone frozen.")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Total params    : {sum(p.numel() for p in model.parameters()):,}")

dummy = torch.randn(2, 15, 3, 224, 224).to(device)  # 15 = 5 slices × 3 planes
with torch.no_grad():
    out = model(dummy)
print(f"Output shape: {out.shape}")


#-----------CHECKER


from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# ── Golden split ──────────────────────────────────────────────────────────────
original_df = pd.read_csv("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
golden_uids = set(original_df[original_df[["ACL","MCL","Medial Meniscus","Lateral Meniscus",
                                            "Medial OA","Lateral OA","PF OA","Effusion",
                                            "Synovitis","Baker's","Contusion","Fracture"]]
                               .notna().any(axis=1)]["StudyInstanceUID"])

golden_mask  = labels_df["StudyInstanceUID"].isin(golden_uids)
val_labels   = labels_df[golden_mask].reset_index(drop=True)
train_labels = labels_df[~golden_mask].reset_index(drop=True)

print(f"Train: {len(train_labels)}  |  Val (golden): {len(val_labels)}")

# ── Loss + optimizer ──────────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()



print("Setup ready ✓")

#---------------------


import time

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            B, S, C, H, W = images.shape

            # Original
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            preds  = torch.sigmoid(logits)

            # TTA — horizontal flip
            flipped = torch.flip(images.view(B*S, C, H, W), dims=[-1]).view(B, S, C, H, W)
            preds  += torch.sigmoid(model(flipped))

            # TTA — rotation +10
            rotated_p = TF.rotate(images.view(B*S, C, H, W), 10).view(B, S, C, H, W)
            preds    += torch.sigmoid(model(rotated_p))

            # TTA — rotation -10
            rotated_n = TF.rotate(images.view(B*S, C, H, W), -10).view(B, S, C, H, W)
            preds    += torch.sigmoid(model(rotated_n))

            preds /= 4  # average 4 augmentations

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    aucs = []
    for i, c in enumerate(CONDITIONS):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        if len(np.unique(y_true)) < 2:
            continue
        aucs.append(roc_auc_score(y_true, y_pred))

    return total_loss / len(loader), np.mean(aucs) if aucs else 0.0

# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_auc    = 0.0
        self.should_stop = False

    def step(self, val_auc: float) -> bool:
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.counter  = 0
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Optimizer — only non-frozen params initially ──────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# ── Training loop ─────────────────────────────────────────────────────────────
best_auc       = 0.0
best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
backbone_unfrozen = False
weighted_criterion = nn.BCEWithLogitsLoss(reduction='none')
for epoch in range(EPOCHS):

    # ── Unfreeze backbone at epoch 5 ─────────────────────────────────────────
    if epoch == 5 and not backbone_unfrozen:
        for param in model.backbone.parameters():
            param.requires_grad = True

     
        
        optimizer = optim.AdamW([
            {"params": model.backbone.parameters(),       "lr": LR * 0.1},
            {"params": model.transformer.parameters(),    "lr": LR},
            {"params": model.plane_embedding.parameters(),"lr": LR},
            {"params": model.classifier.parameters(),     "lr": LR},
        ], weight_decay=1e-4)

         
       

        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        backbone_unfrozen = True
        print("\nBackbone unfrozen — differential LR applied ✓")
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model.train()
    train_loss = 0.0
    t0         = time.time()
    
    for batch_idx, (images, labels, study_uids) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        is_golden = torch.tensor(
            [uid in golden_uids for uid in study_uids],
            dtype=torch.float32, device=device
        )
        weights = 1.0 + 7.0 * is_golden
        loss = weighted_criterion(logits, labels)
        loss = (loss.mean(dim=1) * weights).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {train_loss/(batch_idx+1):.4f}")

    scheduler.step()

    val_loss, val_auc = evaluate(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    print(f"\nEpoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val AUC: {val_auc:.4f} | "
          f"Time: {elapsed:.0f}s")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({
            "epoch"      : epoch + 1,
            "model_state": model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "val_auc"    : val_auc,
        }, best_ckpt_path)
        print(f"  ✓ Best model saved — AUC: {val_auc:.4f}")

    if early_stopping.step(val_auc):
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

print(f"\nTraining done. Best Val AUC: {best_auc:.4f}")