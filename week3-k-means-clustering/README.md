# Unsupervised Clustering of Optical Fibre OTDR Signals

โปรเจกต์ Week 3 สำหรับวิเคราะห์สัญญาณ Optical Time-Domain Reflectometer (OTDR)
ด้วย K-Means clustering โดยใช้ชุดข้อมูล
[Optical Fibre - Fault Detection](https://www.kaggle.com/datasets/yogi2727/optical-fibre-fault-detection)
จาก Kaggle

## Project structure

```text
week3-k-means-clustering/
├── pyproject.toml
├── config.json
├── src/optical_fibre_clustering/
│   ├── download.py
│   └── notebook.py
├── tests/
│   ├── test_download.py
│   └── test_notebook.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── optical_fibre_kmeans_clustering.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
└── report/
```

ไฟล์ต้นฉบับใน `data/raw/` ถือเป็น immutable source data: ไม่แก้ไขด้วยมือ และไม่ commit
ไฟล์ CSV ขนาดใหญ่เข้า Git ตัวดาวน์โหลดจะสร้าง metadata ที่มี URL, เวลา download,
ขนาดไฟล์ และ SHA-256 checksum เพื่อให้ตรวจสอบที่มาของข้อมูลได้ โปรเจกต์ pin
Kaggle dataset ที่ version 1 เพื่อให้ทุกคนใช้ข้อมูลชุดเดียวกัน

## Environment setup

โปรเจกต์นี้ใช้ Python 3.12 เพื่อหลีกเลี่ยงปัญหา compatibility กับ Python 3.14

```bash
cd week3-k-means-clustering
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

ตรวจสอบ environment:

```bash
python --version
python -c "import pandas, sklearn, kagglehub; print('environment OK')"
pytest
```

## Download the Kaggle dataset

ดาวน์โหลดเฉพาะไฟล์ `OTDR_data.csv` ไปที่ `data/raw/`:

```bash
download-otdr-data
```

หรือเรียกผ่าน Python module:

```bash
python -m optical_fibre_clustering.download
```

ถ้ามีไฟล์อยู่แล้ว คำสั่งจะใช้ไฟล์เดิมโดยไม่เขียนทับ หากต้องการดาวน์โหลดใหม่จริง ๆ:

```bash
download-otdr-data --force
```

ข้อมูลสาธารณะทั่วไปดาวน์โหลดได้โดยไม่ต้อง login หาก Kaggle ขอ authentication หรือ
consent ให้สร้าง API token ที่ [Kaggle API settings](https://www.kaggle.com/settings/api)
แล้วเก็บ token ตามคำแนะนำของ
[KaggleHub](https://github.com/Kaggle/kagglehub#authenticate) ห้าม commit token ลง Git

## Build and run the analysis notebook

สร้าง notebook ใหม่จาก source template:

```bash
build-kmeans-notebook
```

จากนั้นรันทุก cell จาก clean kernel และเก็บ output ไว้ใน notebook:

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/optical_fibre_kmeans_clustering.ipynb
```

เปิดแบบ interactive:

```bash
jupyter lab notebooks/optical_fibre_kmeans_clustering.ipynb
```

Notebook ที่ execute แล้วอยู่ที่
[`notebooks/optical_fibre_kmeans_clustering.ipynb`](notebooks/optical_fibre_kmeans_clustering.ipynb)
และ HTML preview อยู่ที่
[`report/optical_fibre_kmeans_clustering.html`](report/optical_fibre_kmeans_clustering.html)

## Implemented analysis

Notebook ทำงานตามลำดับนี้:

1. ตรวจ rows, columns, feature types และ missing values
2. เตรียม numeric/categorical features และใช้ `StandardScaler`
3. ทดลอง K-Means สำหรับ `k = 2..10`
4. เปรียบเทียบ Silhouette Score และ Davies-Bouldin Score
5. ลดมิติด้วย PCA แล้วแสดง cluster ในกราฟ 2D
6. สรุปผลและ render เป็น HTML preview เพื่อเตรียมทำ PDF report ในขั้นต่อไป

ชุดข้อมูล version 1 มี 125,832 แถวและ 36 คอลัมน์ โดย feature สำหรับ clustering
เบื้องต้นคือ `SNR` และ `P1` ถึง `P30` ส่วน `Class` เป็น label ที่ทราบอยู่แล้ว จึงกันออกจาก
K-Means เช่นเดียวกับ `Position`, `Reflectance` และ `loss`; คอลัมน์เหล่านี้ใช้เฉพาะตอน
ตีความผลภายหลังเพื่อรักษาหลัก unsupervised learning

ผลหลักหลังลบ exact duplicates 6,824 แถว:

- ข้อมูลที่ใช้ clustering: 119,008 แถว × 31 features
- Silhouette Score สูงสุด: `K=2`, score `0.2420`
- Davies–Bouldin Score ต่ำสุด: `K=3`, score `1.5017`
- โมเดลหลัก `K=2`: cluster sizes 74,436 และ 44,572 แถว
- PCA สองแกนอธิบาย variance รวมประมาณ 60.0%
- Post-hoc comparison: ARI `0.0973`, NMI `0.1873`; clusters เป็น broad signal
  families และไม่ตรงกับ fault classes ทั้งแปดแบบหนึ่งต่อหนึ่ง

> หมายเหตุ: Kaggle ระบุ license ของ dataset นี้เป็น `Unknown` จึงไม่ควรนำไฟล์ CSV
> ไป redistribute หรือ commit เข้า repository; ให้ผู้ใช้แต่ละคนดาวน์โหลดจากหน้าต้นทาง
