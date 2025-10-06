# DigitalPulse AI  

Bu proje, sosyal medya yorumlarını analiz etmek amacıyla geliştirilmiş uçtan uca bir yapay zeka pipeline’ıdır. Sistem; veri ön işleme, konu eşleştirme, argüman sınıflandırma, özetleme (conclusion generation) ve event-driven entegrasyon adımlarını içermektedir.  

## 🚀 Özellikler  
- Preprocessing: Veri temizleme, dil filtresi (EN), token uzunluğu ve duplicate kontrolü.  
- Topic Matching: SentenceTransformers tabanlı retriever modelleri (all-mpnet-base-v2, e5-large-v2, multi-qa-mpnet-base-dot-v1) + FAISS GPU desteği.  
- Argument Classification: DeBERTa-v3-base modeli ile dört sınıf (Claim, Evidence, Counterclaim, Rebuttal) için fine-tuning.  
- Conclusion Generation: Mistral-7B-Instruct-v0.2 ile topic bazlı stance-aware kısa özetler.  
- Evaluation: ROUGE, BLEU, BERTScore ve stance accuracy metrikleri.  
- Event-driven Integration: gRPC server, RabbitMQ ve PostgreSQL tabanlı gerçek zamanlı işleme.  

## 📊 Özet Sonuçlar  
- Argument Classifier: Accuracy ≈ %80 (özellikle Claim/Evidence sınıflarında güçlü performans)  
- Event-driven Entegrasyon: 200 kayıtlık örnek başarıyla sınıflandırılmış, eşleştirilmiş ve özetlenmiştir.  

## 📂 Proje Yapısı  
```
├── data/               # Raw, interim ve processed veri setleri
├── src/                # Pipeline scriptleri (preprocessing, topic_matching, arg_classifier, conc_generator)
├── results/            # Çıktılar (metrics, predictions, evaluation, integration results)
├── inference/          # gRPC entegrasyonu için inference scriptleri
├── configs/            # Parametre ayarları (YAML)
├── requirements.txt    # Python bağımlılıkları
├── environment.yml     # Conda ortam konfigürasyonu
└── docker-compose.yml  # Servislerin (gRPC, RabbitMQ, PostgreSQL) entegrasyonu
```

## ⚙️ Kullanım  
1. Preprocessing  
   ```bash
   python src/preprocessing.py
   ```  
2. Topic Matching  
   ```bash
   python src/topic_matching.py
   ```  
3. Argument Classifier  
   ```bash
   python src/argument_classifier.py
   ```  
4. Conclusion Generation  
   ```bash
   python src/conclusion_generator.py
   ```  
5. Evaluation  
   ```bash
   python src/evaluation.py
   ```  
6. Event-driven Run  
   ```bash
   python -m integration.server  
   python -m integration.publisher  
   python -m integration.event_handler  
   ```  

## ⚙️ Konfigürasyon Dosyaları  
- requirements.txt → Pipeline’ın çalışması için gerekli Python kütüphaneleri  
- environment.yml → Conda ortamı için tüm bağımlılıkların sürümleri  
- configs/*.yml → Model, batch size, top-k değerleri, temperature gibi parametreler  
- docker-compose.yml → RabbitMQ, PostgreSQL ve gRPC servislerinin tek komutla ayağa kaldırılması için  

## 📌 Genel Değerlendirme  
Bu pipeline, production-ready demo seviyesinde geliştirilmiş olup; yüksek hacimli sosyal medya yorumlarını otomatik olarak işleyerek konu eşleştirme, argüman sınıflandırma ve stance bazlı özet üretimi yapabilmektedir. GCP altyapısı üzerinde ölçeklenebilir, güvenilir ve entegrasyona uygun bir çözüm sunmaktadır.  
