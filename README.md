# Continuous Learning

Continuous Learning의 주요 요소인 Feature Extraction, Finetuning, Knowledge Distillation(KD), Learning Without Forgetting(LWF)기법으로 cifiar10 성능 실험결과에 대해서 다룬다. 코드는 Pytorch 기반으로 작성 되었다.

## 개념

본 실험에서는 편의를 위해 Teacher Network(Pre-trainined)로 VGG16 모델을 사용하였고, Student Network은 VGG16보다 얕은 layer와 channel를 가진 모델을 임의로 생성을 하였다. 모델에 대한 코드는 model.py에서 확인 가능하다.

1. Feature Extraction

2. Finetuning

3. Knowledge Distillation(KD) 

4. Learning Without Forgetting(LWF)


