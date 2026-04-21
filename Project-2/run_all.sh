#!/bin/bash
# Tüm modelleri sırayla eğitir. Her adımın çıktısı ilgili outputs/ klasöründeki
# run.log dosyasına yazılır; ayrıca ekrana da basılır.
#
# Kullanım:  bash run_all.sh
#
# Herhangi bir adım başarısız olursa script durur (set -e).

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo " Eğitim pipeline'ı başlıyor"
echo " Başlangıç: $(date)"
echo "=========================================="

run_step() {
    local name="$1"
    local cmd="$2"
    local log="$3"
    echo ""
    echo ">>> $name"
    echo ">>> Komut: $cmd"
    echo ">>> Log  : $log"
    echo ">>> Başlangıç: $(date)"
    bash -c "$cmd" 2>&1 | tee "$log"
    echo ">>> Bitiş    : $(date)"
}

run_step "1/7 Model 1 — LeNet Temel" \
    "python3 model1_lenet_basic/train.py" \
    "model1_lenet_basic/outputs/run.log"

run_step "2/7 Model 2 — LeNet İyileştirilmiş" \
    "python3 model2_lenet_improved/train.py" \
    "model2_lenet_improved/outputs/run.log"

run_step "3/7 Model 3 — AlexNet (pretrained fine-tune)" \
    "python3 model3_alexnet/train.py" \
    "model3_alexnet/outputs/run.log"

run_step "4/7 Model 4 — Özellik çıkarımı (.npy)" \
    "python3 model4_hybrid/extract_features.py" \
    "model4_hybrid/outputs/run_extract.log"

run_step "5/7 Model 4a — SVM eğitimi" \
    "python3 model4_hybrid/train_svm.py" \
    "model4_hybrid/outputs/run_svm.log"

run_step "6/7 Model 4b — Random Forest eğitimi" \
    "python3 model4_hybrid/train_rf.py" \
    "model4_hybrid/outputs/run_rf.log"

run_step "7/7 Karşılaştırma tablosu" \
    "python3 comparison/build_comparison.py" \
    "comparison/run.log"

echo ""
echo "=========================================="
echo " TÜM ADIMLAR TAMAMLANDI"
echo " Bitiş: $(date)"
echo "=========================================="
