#!/bin/bash


# echo "================================================================================================="
# echo "Experiment 1"
echo "-------------------------------------------------------------------------------------------------"
echo "Exp1 ResNet34"
python3 Experiment1_Test.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip

# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp1 ResNet34+RF(MSE)"
# python3 Experiment1_Test.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Reg --model1_flip --model2_flip



echo "================================================================================================="
echo "Experiment 3"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 ResNet34"
# python3 Experiment3.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 ResNet34+RF(Gini)"
# python3 Experiment3.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 LQReg"
# python3 Experiment3.py --model1 QReg --model2 None --PredType1 Reg --PredType2 Reg
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 ResNet34+RF(MSE)"
# python3 Experiment3.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Reg --model1_flip --model2_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 [ResNet34, LQReg]"
# python3 Experiment3.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip --EnsemblePred
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 [ResNet34+RF(Gini), LQReg]"
# python3 Experiment3.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip --EnsemblePred
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 ResNet34 DA"
# python3 Experiment3.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip --dataAugment --DA_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 ResNet34+RF(Gini) DA"
# python3 Experiment3.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip  --dataAugment --DA_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 LQReg DA"
# python3 Experiment3.py --model1 QReg --model2 None --PredType1 Reg --PredType2 Reg  --dataAugment --DA_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 ResNet34+RF(MSE) DA"
# python3 Experiment3.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Reg --model1_flip --model2_flip  --dataAugment --DA_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 [ResNet34, LQReg] DA"
# python3 Experiment3.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip --EnsemblePred  --dataAugment --DA_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp3 [ResNet34+RF(Gini), LQReg] DA"
# python3 Experiment3.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip --EnsemblePred  --dataAugment --DA_flip
#
#
echo "================================================================================================="
echo "Experiment 4"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 ResNet34"
# python3 Experiment4.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 ResNet34+RF(Gini)"
# python3 Experiment4.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 LQReg"
# python3 Experiment4.py --model1 QReg --model2 None --PredType1 Reg --PredType2 Reg
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 ResNet34+RF(MSE)"
# python3 Experiment4.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Reg --model1_flip --model2_flip
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 [ResNet34, LQReg]"
# python3 Experiment4.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip --EnsemblePred
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 [ResNet34+RF(Gini), LQReg]"
# python3 Experiment4.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip --EnsemblePred
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 ResNet34 DA knowMC"
# python3 Experiment4.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip --dataAugment --DA_flip --knowMC
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 ResNet34+RF(Gini) DA knowMC"
# python3 Experiment4.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip  --dataAugment --DA_flip --knowMC
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 LQReg DA knowMC"
# python3 Experiment4.py --model1 QReg --model2 None --PredType1 Reg --PredType2 Reg  --dataAugment --DA_flip --knowMC
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 ResNet34+RF(MSE) DA knowMC"
# python3 Experiment4.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Reg --model1_flip --model2_flip  --dataAugment --DA_flip --knowMC
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 [ResNet34, LQReg] DA knowMC"
# python3 Experiment4.py --model1 ResNet34 --model2 CNN --PredType1 Cl --PredType2 Cl --model1_flip --EnsemblePred  --dataAugment --DA_flip --knowMC
# echo "-------------------------------------------------------------------------------------------------"
# echo "Exp4 [ResNet34+RF(Gini), LQReg] DA knowMC"
# python3 Experiment4.py --model1 ResNet34 --model2 RF --PredType1 Cl --PredType2 Cl --model1_flip --model2_flip --EnsemblePred  --dataAugment --DA_flip --knowMC
