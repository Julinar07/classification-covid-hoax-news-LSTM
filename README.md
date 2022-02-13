# Berita COVID-19 Indonesia
Sumber : https://covid19.go.id/p/berita

Data utama : [data.csv](/data.csv)

snapshot(raw) : [snapshot.json](/snapshot.json) --> pakai yg ini kalau mau preprocessing sendiri

## Notes 
- Cleaning yg dilakukan disini
    - Menghapus substring `"[SALAH]"` dan menghapus quotes pada judul tersebut
    - Menghilangkan data duplikat dan reset index
- Mungkin masih ada whitespace pada datanya, bisa dibersihkan sendiri

