#!/bin/bash

# make sure your environment is activated
# LMH.local:$ venv ml

echo ""
echo "–––––––––DEEP LEARNING FOR ANALYSING IMMUNE CELL INTERACTIONS–––––––––"
echo "Running evaluation --> reconstruction + low-dimensional visualisation + regression"
echo "Figures will be saved in /l4proj/data/evaluation/"
echo "/evaluation/"
echo "           autoencoder/ for autoencoder reconstruction"
echo "           clustering/ for t-sne/UMAP projection"
echo "           regression/ for t-sne/UMAP projection"

echo ""
echo "---------Evaluating unmasked datasets...---------"

echo ""
echo "DS1: balanced dataset"
echo "Warning: the model was trained on part of this data and only the last 10000 images should be used"
#python main.py --file "/Volumes/TARDIS/CK19_full.npz" --weights "../data/processed/decoder_weights.h5" "../data/processed/encoder_weights.h5" --label "balanced"

echo ""
echo "DS2: DMSO dataset"
echo "Warning: the model was trained on a small part of this data"
#python main.py --file "/Volumes/TARDIS/DMSO_unmodified.npz" --weights "../data/processed/decoder_weights.h5" "../data/processed/encoder_weights.h5" --label "dmso"

echo ""
echo "DS3: balanced 2-category dataset (unseen)"
#python main.py --file "/Volumes/TARDIS/CK22_full.npz" --weights "../data/processed/decoder_weights.h5" "../data/processed/encoder_weights.h5" --label "CK22"

echo ""
echo "---------Evaluating masked datasets...---------"

echo ""
echo "DS1: masked balanced dataset"
echo "Warning: the model was trained on part of this data and only the last 10000 images should be used"
#python main.py --file "/Volumes/TARDIS/CK19_full.npz" --weights "../data/processed/decoder_masked_weights.h5" "../data/processed/encoder_masked_weights.h5" --mask --label "balanced"

echo ""
echo "DS2: masked DMSO dataset"
echo "Warning: the model was trained on a small part of this data"
#python main.py --file "/Volumes/TARDIS/DMSO_unmodified.npz" --weights "../data/processed/decoder_weights.h5" "../data/processed/encoder_weights.h5" --mask --label "dmso"

echo ""
echo "DS3: masked balanced 2-category dataset (unseen)"
#python main.py --file "/Volumes/TARDIS/CK22_full.npz" --weights "../data/processed/decoder_weights.h5" "../data/processed/encoder_weights.h5"  --mask --label "CK22"

echo ""
echo "–––––––––EVALUATION COMPLETED–––––––––"
echo ""
