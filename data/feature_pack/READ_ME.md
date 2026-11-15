# feature_paketi_20261029 — kısa açıklama

RNA–RNA etkileşim hedef matrisi (RNAInter, GRCh38, 250 kb bin, TSS çapası) için aday feature verisi.
Ayrıntılı açıklama ve makale kaynakları ayrı belgede: "RNA–RNA feature paketi: açıklamalar".

## Paketteki dosyalar

| Dosya | İçerik |
|---|---|
| `READ_ME.md` | Bu açıklama |
| `bin_features.xlsx` | Bin başına özellik tablosu: 12.366 bin (GRCh38, 250 kb), 80 özellik sütunu, 5 sayfa |
| `pair_features.xlsx` | İki bine birden ait çift özelliği (`kmer7_revcomp_cosine`); her satır bir bin çifti, 2 sayfa |
| `feature_tiers.csv` | `bin_features.xlsx`'teki her sütun için ne ölçtüğü, tür, düzey, hücre hattı, kademe, temsilci olup olmadığı, neyle tekrarlı olduğu ve dayanak makale |

## `bin_features.xlsx`

Her satır bir bin, yani genomda 250 kb'lik bir bölge. Her sayfanın ilk 5 sütunu kimliktir, özellik değildir:
`chrom`, `start`, `end`, `bin` (genom boyunca 0–12365), `matrix_bin_index` (kromozom içinde 0'dan başlayan sıra; hedef matrisin satır/sütun indeksiyle aynı).
Genler, hedef matristeki gibi TSS'lerinin düştüğü bine atandı.

| Sayfa | Özellik sütunu | İçerik |
|---|---|---|
| `direct` | 13 | Doğrudan özellikler: düşük karmaşıklıklı tekrar (LCR) oranları, RNA yönünde G/C/GA/UC tekrar alt tipleri, 3′UTR'de aynı/zıt yönlü Alu sayıları |
| `indirect_tier1_control` | 10 | Kontrol: gen sayıları, transkript uzunluğu, miRNA sayısı, LCR paydası |
| `indirect_tier2_context` | 26 | Destekli bağlam: ekspresyon (GM12878 ve HeLa), GM12878 miRNA ifadesi |
| `indirect_tier3_partial` | 12 | Kısmi destek: CDS/UTR uzunlukları, m6A, HeLa hücre içi konum oranları |
| `indirect_tier4_explore` | 19 | Keşif / teknik: gen uzunluğu, izoform, büyük intron, RNA-seq kapsama/iplik ölçüleri, GC oranı, HeLa eşleşen gen sayısı |

Dolaylı sayfalarda temsilci sütunlar başta gelir. Aynı bilginin varyantlarından (rep1/rep2, ham/log TPM, TPM eşikleri)
yalnızca biri `feature_tiers.csv`'de `representative = yes` olarak işaretli.

Boş hücre = o bin için bilgi yok (ör. binde ilgili türde gen yok veya veri eşlenmedi); sıfır değildir.

### Sonradan eklenen sütun: `mirna_expr_sum_log1p_cpm` (`indirect_tier2_context`)

- **Ne ölçer:** TSS'i bu binde olan miRNA genlerinin GM12878'deki toplam ifadesi: log(1 + CPM toplamı), iki replikatın ortalaması.
- **Kaynak:** ENCODE GM12878 microRNA-seq, ENCSR770HBF (dosyalar ENCFF343RGE ve ENCFF652ODI, GRCh38, GENCODE V29). Sayımlar ipliğe özgü sütundan alındı ve her replikat kendi toplamına göre CPM'e çevrildi.
- **Bine yerleştirme:** 1.879 miRNA geninin 1.642'si hedefin eşleme dosyasındaki (CS401 `symbol_to_bin`) bine, kalan 237'si GENCODE V29 gen TSS'ine göre aynı 250 kb kuralıyla yerleştirildi.
- **Değerler:** 1.347 binde değer var; binde miRNA geni yoksa boş. İki replikatın bin düzeyinde uyumu Spearman 0,92; `mirna_mature_count` ile Spearman 0,49.
- **Neden:** `mirna_mature_count` binde kaç miRNA olduğunu söyler, bu sütun ne kadar ifade edildiğini. Seed eşleşmesine dayanmaz, bu yüzden RNAInter'daki tahmine dayalı miRNA–mRNA kayıtlarının kuruluş kuralını tekrar etmez.
- **Sınırlama:** ENCODE hattı birden fazla lokusa hizalanan okumaları saymaz; aynı olgun diziyi taşıyan, birden çok lokustan gelen miRNA'lar olduğundan düşük görünebilir.

## `pair_features.xlsx` — çift özellikleri

### Neden ayrı bir dosya?

`bin_features.xlsx`'teki her değer **tek bir bine** aittir (ör. "bin 174'ün ekzonlarının %1,5'i LCR"); bu yüzden tablo 12.366 satırdır ve her sütun bir özelliktir.
`pair_features.xlsx`'teki değerler ise **iki bine birden** aittir (ör. "bin 174'teki RNA'lar ile bin 178'deki RNA'lar birbirine ne kadar tamamlayıcı").
Böyle bir değer tek bir binin satırına yazılamaz; bu yüzden her satırın bir bin çiftini temsil ettiği ayrı bir tablo gerekir.
Bu özellik bin tablosundaki sütunlardan türetilemez; doğrudan RNA dizilerinden hesaplandı.

### Tablonun yapısı

Her satır bir bin çifti:

| Sütun | Anlamı |
|---|---|
| `chrom` | Kromozom |
| `bin_i`, `bin_j` | İki binin kromozom içindeki sırası; `bin_features.xlsx`'teki `matrix_bin_index` ile aynı |
| `value` | O çiftin değeri |

- **Yalnızca aynı kromozomdaki bin çiftleri var.** Kromozomlar arası çiftler hesaplanmadı.
- **Değer simetrik:** (i, j) ile (j, i) aynıdır; her çift bir kez yazıldı (`bin_i ≤ bin_j`). `bin_i = bin_j` satırları binin kendi içindeki değeridir.
- **Yalnızca sıfır olmayan çiftler var.** Tabloda olmayan bir çiftin değeri 0'dır: binlerden birinde mRNA yoktur veya hiç eşleşme bulunmamıştır.
- **Değerler 6 ondalık basamağa yuvarlandı.**

### İçindeki özellik

**Çıkarılan özellik:** `mirna_seed_sites_3utr` (miRNA seed sitelerinin 3′UTR'lerde sayımı) paketten çıkarıldı. Hedefteki miRNA–mRNA kayıtlarının %64'ü TargetScan, miRanda gibi seed eşleşmesine dayanan hesaplamalı tahminler; bu özellik girdi olursa hedefin kurulduğu kuralı tekrar eder (hedef sızıntısı). Veri silinmedi, `outputs/feature_paketi_disinda/mirna_seed_sites_3utr.xlsx` dosyasında saklanıyor; hedef yalnızca deneysel kayıtlara indirilirse yeniden değerlendirilebilir. miRNA için pakette seed kullanmayan iki sütun var: `mirna_mature_count` ve `mirna_expr_sum_log1p_cpm`.

**`kmer7_revcomp_cosine`** (sayfalar `kmer7_revcomp_cosine_1`: chr1–9, 740.799 satır; `kmer7_revcomp_cosine_2`: chr10–Y, 369.770 satır)

- **Ne ölçer:** İki bindeki mRNA dizilerinin birbirine ne kadar tamamlayıcı olduğu: bir bindeki mRNA'ların 7 harflik dizi parçalarının (7-mer)
  dağılımı ile diğer bindekilerin ters-tamamlayıcı 7-mer dağılımı karşılaştırılır.
- **Değer:** 0 ile 1 arası benzerlik (kosinüs). Sıfır olmayanlarda medyan 0,34, en büyük 0,85.
- **Kaynak:** NCBI GRCh38.p14 mRNA dizileri (gen başına tek temsilci transkript); gen–bin eşlemesi hedef matrisi kuran CS401 eşleme dosyasından.
- **Dayanak:** RIME (Setti 2026) tamamlayıcılığı 7-mer'leri ters-tamamlayıcılarıyla eşleyerek inceler; RIblast uzun RNA'larda benzer bir arama yapar.
  Bin düzeyinde bu benzerlik ölçüsü bizim uyarlamamızdır; makalede birebir yoktur.
- **Sınırlama:** GC içeriğiyle karışabilir. Kademe 4 (keşif).
- **Neden iki sayfa:** 1,1 milyon satır, Excel'in sayfa başına satır sınırını (1.048.576) aşar.

### Bin tablosuyla ilişkisi

- `chrom` + `bin_i` / `bin_j`, `bin_features.xlsx`'teki `chrom` + `matrix_bin_index` ile aynı binleri gösterir.
- Tamamlayıcılığın kendisi bin tablosunda yoktur; yalnızca bu dosyada.

## Doğrudan ve dolaylı

- **Doğrudan** sütunlar baz eşleşmesi potansiyeliyle ilgili dizi içeriğini ölçer. Bunlar **RRI ile ilişkili RNA özellikleridir**:
  tek bir binin RNA'larının eşleşmeye yatkınlığını gösterir, hangi RNA ile etkileşeceğini tek başına söylemez.
  Hücrede gerçekleşmiş RNA–RNA temasının kanıtı da değildir.
- **Dolaylı** sütunlar eşleşmeyle ilgili değildir; bolluk, gen sayısı, uzunluk, yapı, konum gibi etkileşimin gözlenme
  olasılığını etkileyen bilgilerdir.
- Bu paket hazırlanmış **aday** feature'lardır; hiçbir sütunun katkısı henüz gösterilmedi.

## Hangi RNA'lar, hangi hücre hattı

- **RNA türü:** `direct` sayfasının tamamı protein-kodlayan genlere (mRNA) dayanır; yalnızca `rmsk_*_genomic_fraction` binin tüm DNA'sına bakar.
  Dolaylı sütunların bir kısmı yalnızca mRNA'ya, bir kısmı bindeki tüm genlere bakar; `gencode_lncRNA_count` lncRNA'ları,
  `mirna_mature_count` olgun miRNA'ları sayar. Yalnızca lncRNA veya miRNA içeren binlerde mRNA'ya dayanan sütunlar boş ya da 0'dır.
- **Hücre hattı** (`feature_tiers.csv` → `cell_line`):
  - **GM12878** (34 sütun): ekspresyon (`rna_seq_*`, `total_rna_seq_*`), RNA-seq sinyali (`polyA_rep2_*`), ekspresyona dayanan splicing sütunları, m6A (`gse65380_*`), miRNA ifadesi (`mirna_expr_sum_log1p_cpm`; ENCODE ENCSR770HBF microRNA-seq, seed eşleşmesine dayanmaz).
  - **HeLa** (8 sütun): `hela_*` (GSE110028 fraksiyon verisi: hücre içi konum oranları ve fraksiyon TPM'leri).
  - **Hücre hattından bağımsız** (38 sütun): genom anotasyonu ve diziden hesaplanan sütunlar (tüm `direct` sayfası, gen sayıları, uzunluklar, intron ölçüleri, miRNA sayısı).
  - RNAInter hedefi tek bir hücre hattından değildir; mRNA–mRNA kayıtlarının çoğu PARIS'ten (HeLa, HEK293T) gelir.
  - GM12878 ve HeLa sütunları iki farklı hücre hattının ölçümleridir; aynı ölçümün kopyası değildir ve değerleri birbirinin yerine geçmez.
    HeLa sütunları, mRNA–mRNA kayıtlarının çoğunu sağlayan PARIS deneylerinin yapıldığı hücre hatlarından biri olduğu için GM12878'e alternatif olarak eklendi.

## `feature_tiers.csv` sütunları

| Sütun | Anlamı |
|---|---|
| `location` | Excel sayfası (`bin_features.xlsx/<sayfa>`) |
| `feature` | Sütun adı |
| `description` | Sütunun ne ölçtüğü (kısa tanım) |
| `type` | `direct` / `indirect` / `id` |
| `level` | `rna (bin)`: bin başına RNA özelliği |
| `cell_line` | GM12878 / HeLa / hücre hattından bağımsız |
| `tier` | Kademe: 1 = önce, 4 = keşif (makale desteğine ve hücre hattı uyumuna göre deneme önceliği) |
| `representative` | Aynı bilginin varyantları arasından seçilen sütun (`yes`) |
| `redundant_with` | Tekrarlı olduğu temsilci sütun |
| `evidence` | Dayanak makale(ler) |

## Hedef verisiyle ilgili not

İnsan mRNA–mRNA kayıtlarının (54.703) %32'si "Microarray" kanıtlı; mikrodizi iki RNA'nın fiziksel temasını ölçmez.
