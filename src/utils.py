import os
from tqdm.auto import tqdm




def preprocess_and_save(study_uid: str, series_map: pd.DataFrame,
                         mri_dir: Path, out_dir: Path) -> bool:
    study_series = series_map[series_map["StudyInstanceUID"] == study_uid]
    study_out    = out_dir / study_uid
    study_out.mkdir(parents = True,exist_ok=True)

    for plane in PLANES:
        row = study_series[study_series["Anatomical_Plane"] == plane]
        if len(row) == 0:
            continue
        series_uid  = row.iloc[0]["SeriesInstanceUID"]
        series_path = mri_dir / study_uid / series_uid
        plane_out   = study_out / plane
        plane_out.mkdir(exist_ok=True)

        if len(list(plane_out.glob("*.png"))) == N_SLICES:
            continue
        try:
            volume = load_dicom_volume(series_path)
            slices = select_center_slices(volume, N_SLICES)
            for i, slc in enumerate(slices):
                img_resized = cv2.resize(slc, (IMG_SIZE, IMG_SIZE),
                                         interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(plane_out / f"slice_{i}.png"), img_resized)
        except Exception as e:
            print(f"Failed {study_uid} / {plane}: {e}")
            return False
    return True

# ── Check if already done ─────────────────────────────────────────────────────
existing = len(list(PREPROCESS_DIR.rglob("*.png")))
if existing >= 4400 * 3 * N_SLICES:
    print(f"Already done — {existing} PNGs found, skipping.")
else:
    print(f"Found {existing} PNGs, running preprocessing...")
    all_uids = labels_df["StudyInstanceUID"].tolist()
    failed   = []
    for uid in tqdm(all_uids, desc="Preprocessing"):
        ok = preprocess_and_save(uid, series_map, MRI_DIR, PREPROCESS_DIR)
        if not ok:
            failed.append(uid)
    print(f"\nDone. Failed: {len(failed)}")
    print(f"Disk used: {sum(f.stat().st_size for f in PREPROCESS_DIR.rglob('*.png'))/1e9:.2f} GB")