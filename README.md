# Pill Project
🗓 프로젝트 소개 : Pill Project</br>
🗓 기간 : 2022.9.24 ~   </br>
🗓 팀원:  [준석](https://github.com/dotdotot)</br>
🗓 리뷰어: [준석](https://github.com/dotdotot)</br></br>

# Yolov5
yolo 커스텀 데이터셋 학습시키기(colab)</br>

1. 드라이브 마운트</br>
<code>
    #드라이브 마운트</br>

    from google.colab import drive

    drive.mount('/content/drive')
</code>

2. 환경 세팅</br>
<code>
    내 구글 드라이브로 이동

    %cd "/content/drive/MyDrive"

    Yolov5 github 레포지토리 clone

    !git clone https://github.com/ultralytics/yolov5.git

    필요한 모듈 설치
    !pip install -U -r yolov5/requirements.txt
</code>
<code>
    import torch

    #파이토치 버전 확인, cuda device properties 확인

    print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
<code>

런타임 -> 런타임 유형 변경에 들어가서 하드웨어 가속기 GPU로 변경</br>
이후 custom 모델에 사용할 데이터 수집하기</br>

3. data.yaml 파일 생성</br>
data.yaml : 모델 학습을 위한 작은 튜토리얼 같은 것 (내 드라이브/dataset 아래에 만들어준다)  

![다운로드](https://user-images.githubusercontent.com/77331459/194784144-00d6d2a6-9074-4eed-9f9b-6cca9decdd79.png)  

파일의 내용은 그림과 같이 적어주면 된다.  

class가 여러개라면 nc의 개수를 class의 개수만큼 지정하고 names 배열 내부에 class 이름을 적어주면 된다.  

즉, nc는 자신이 학습시키고자 하는 클래스의 수(number)고  

names에는 그 클래스의  이름을 배열로 적어주면 됨  

여러개일 경우 ['class1','class2'] 처럼..    



4.  labels/images 폴더 정리해주기  

2번에서 만들어놨던 내 데이터셋들을 모두 정리해줘야한다.  

images/train 에는 훈련시키고자 하는 image들을 넣고  

images/val 에는 validation에 사용되는 image들,   

labels/train  훈련시키는 image의 바운딩 박스 정보가 저장된 txt파일들  

labels/val 에는  validation에 사용되는  image의 바운딩 박스 정보 txt파일들 을 모두 업로드 해준다.  


5. 모델 선택하기  

![다운로드 (1)](https://user-images.githubusercontent.com/77331459/194784149-35ee09f9-91a9-42a0-917b-6b39c85f147d.png)   


yolov5/models 에 여러 파일들이 있다. 그 중 하나 선택하여 해당 파일 내용중 nc 를 자신이 학습시키고자 하는 클래스 개수로 바꾼다.  

![다운로드 (2)](https://user-images.githubusercontent.com/77331459/194784150-8db0c7dc-515d-4467-8408-dd77a975670a.png)    



6. training 시키기 !!  

training 시키기 전에 항상 이 코드를 실행해준다  

<code>
    !pip install -U PyYAML
</code>  

코드를 통해 yolov5 디렉토리로 이동  

<code>
    %cd /content/drive/My\ Drive/yolov5
</code>  

!python train.py --img 640 --batch 30 --epochs 100 --data /content/drive/My\ Drive/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name [내가 정한 폴더명]  

를 실행하면 훈련이 시작된다.  

훈련이 시작되고 완료되면 어느 폴더에 .pt파일이 생성이 되었는지 확인가능하다.  
  
  
7. yolov5 실행  
cd /content/drive/MyDrive/yolov5 코드를 실행해 .pt파일이 존재하는 위치로 이동한다  
#사용  

!python detect.py --img 640 --weights "/content/yolov5/runs/train/coustomYolov5m/weights/best.pt" --source "/content/drive/MyDrive/testImages"  



# Color

# Shape

# String

# 사용한 라이브러리
* yolov5

color에서 사용한 라이브러리
* numpy
* cv2
* KMeans
* matplotlib
* PIL
* os

colorCss에서 사용한 라이브러리
* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* os
* PIL
* shutil

shape에서 사용한 라이브러리
* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* os
* PIL
* shutil

string에서 사용한 라이브러리
* Image
* pytesseract

# Commit 규칙
> 커밋 제목은 최대 50자 입력 </br>
본문은 한 줄 최대 72자 입력 </br>
Commit 메세지 </br>

🪛[chore]: 코드 수정, 내부 파일 수정. </br>
✨[feat]: 새로운 기능 구현. </br>
🎨[style]: 스타일 관련 기능.(코드의 구조/형태 개선) </br>
➕[add]: Feat 이외의 부수적인 코드 추가, 라이브러리 추가 </br>
🔧[file]: 새로운 파일 생성, 삭제 시 </br>
🐛[fix]: 버그, 오류 해결. </br>
🔥[del]: 쓸모없는 코드/파일 삭제. </br>
📝[docs]: README나 WIKI 등의 문서 개정. </br>
💄[mod]: storyboard 파일,UI 수정한 경우. </br>
✏️[correct]: 주로 문법의 오류나 타입의 변경, 이름 변경 등에 사용합니다. </br>
🚚[move]: 프로젝트 내 파일이나 코드(리소스)의 이동. </br>
⏪️[rename]: 파일 이름 변경이 있을 때 사용합니다. </br>
⚡️[improve]: 향상이 있을 때 사용합니다. </br>
♻️[refactor]: 전면 수정이 있을 때 사용합니다. </br>
🔀[merge]: 다른브렌치를 merge 할 때 사용합니다. </br>
✅ [test]: 테스트 코드를 작성할 때 사용합니다. </br>