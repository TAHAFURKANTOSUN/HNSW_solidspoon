import numpy as np
import faiss

# 1. Rastgele veri kümesi oluşturalım (Örn: 10.000 tane 2 boyutlu vektör)
d = 2                           # Vektör boyutu (Dimension)
nb = 10000                      # Veri tabanındaki toplam vektör sayısı
nq = 1                          # Aramak istediğimiz vektör sayısı

np.random.seed(42)              # Sonuçların her seferinde aynı çıkması için
xb = np.random.random((nb, d)).astype('float32') # Veri tabanı vektörleri
xq = np.random.random((nq, d)).astype('float32') # Aramak istediğimiz sorgu vektörü

# 2. HNSW İndeksini Oluşturma
# IndexHNSWFlat(boyut, komşuluk_sayısı)
# M = 16: Her düğümün grafik mimarisinde sahip olacağı maksimum bağlantı (komşu) sayısıdır.
M = 16 
index = faiss.IndexHNSWFlat(d, M)

# 3. Vektörleri Grafiğe Ekleme (HNSW katmanları otomatik inşa edilir)
print("İndeks eğitiliyor ve veriler ekleniyor...")
index.add(xb)
print(f"Toplam eklenen vektör sayısı: {index.ntotal}")

# 4. En Yakın Komşu Araması (Search)
k = 3 # En yakın kaç komşuyu bulmak istiyoruz?
distances, indices = index.search(xq, k)

# 5. Sonuçları Ekrana Yazdırma
print("\n--- Arama Sonuçları ---")
print(f"Sorgu Vektörümüz: {xq[0]}")
for i in range(k):
    print(f"{i+1}. En Yakın Vektör İndeksi: {indices[0][i]} | Mesafe (L2): {distances[0][i]:.4f} | Vektör: {xb[indices[0][i]]}")
