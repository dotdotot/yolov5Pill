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

img: 입력 이미지 크기  

batch: 배치 크기  

epochs: 학습 epoch 수 (참고: 3000개 이상이 일반적으로 사용된다고 한다...)  

data: data.yaml 파일 경로  

cfg: 모델 구성 지정  


weights: 가중치에 대한 사용자 정의 경로를 지정합니다(참고: Ultraalytics Google Drive 폴더에서 가중치를 다운로드할 수 있습니다).  

name: 모델이 저장 될 폴더 이름  

nosave: 최종 체크포인트만 저장  

cache: 더 빠른 학습을 위해 이미지를 캐시  
  


!python train.py --img 640 --batch 30 --epochs 100 --data /content/drive/My\ Drive/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name [내가 정한 폴더명]  

를 실행하면 훈련이 시작된다.  

훈련이 시작되고 완료되면 어느 폴더에 .pt파일이 생성이 되었는지 확인가능하다.  
  
  
7. yolov5 실행  
cd /content/drive/MyDrive/yolov5 코드를 실행해 .pt파일이 존재하는 위치로 이동한다  
#사용  

!python detect.py --img 640 --weights "/content/yolov5/runs/train/coustomYolov5m/weights/best.pt" --source "/content/drive/MyDrive/testImages"  



# Color  
CNN(Convolutional Neural Networks)  
CNN(Convolutional Neural Networks)은 수동으로 특징을 추출할 필요 없이 데이터로부터 직접 학습하는 딥러닝을 위한 신경망 아키텍처  
특징 추출 영역은 합성곱층(Convolution layer)과 풀링층(Pooling layer)을 여러 겹 쌓는 형태(Conv+Maxpool)로 구성되어 있으며, 이미지의 클래스를 분류하는 부분은 Fully connected(FC) 학습 방식으로 이미지를 분류  
![images_eodud0582_post_00e763c2-8f36-44e9-9303-7a710256d8c9_image](https://user-images.githubusercontent.com/77331459/194784494-53df4a7a-f72a-498b-b6c7-cfe5d74ac9bf.png)
CNN 알고리즘의 구조  
  
CNN은 주로 이미지나 영상 데이터를 처리할 때 쓰이는데, 영상에서 객체, 얼굴, 장면 인식을 위한 패턴을 찾을 때 특히 유용하며, 오디오, 시계열, 신호 데이터와 같이 영상 이외의 데이터를 분류하는 데도 효과적  
  
과정 :  
1. 데이터셋 준비  
2. 경로 지정 및 데이터 살펴보기  
3. 이미지 데이터 전처리  
4. 모델 구성  
5. 모델 학습  
6. 테스트 평가  
7. 모델 저장  
  
1. 데이터셋 준비  
train, validation, test 폴더를 생성  
  
경로 지정  
<code>
    # 기본 경로  
    base_dir = 'C:\\vsCode\PillProject\image\color\\'  
    train_dir = os.path.join(base_dir, 'train')  
    validation_dir = os.path.join(base_dir, 'validation')  
    test_dir = os.path.join(base_dir, 'test')  
      
    # 훈련용 이미지 경로  
    train_red_dir = os.path.join(train_dir, 'red')  
    train_green_dir = os.path.join(train_dir, 'green')  
    train_blue_dir = os.path.join(train_dir, 'blue')  
    train_orange_dir = os.path.join(train_dir, 'orange')  
    train_white_dir = os.path.join(train_dir, 'white')  
      
    # 검증용 이미지 경로  
    validation_white_dir = os.path.join(validation_dir, 'white')  
    validation_red_dir = os.path.join(validation_dir, 'red')  
    validation_green_dir = os.path.join(validation_dir, 'green')  
    validation_orange_dir = os.path.join(validation_dir, 'orange')  
    validation_blue_dir = os.path.join(validation_dir, 'blue')  
      
    # 테스트용 이미지 경로  
    test_white_dir = os.path.join(test_dir, 'white')  
    test_red_dir = os.path.join(test_dir, 'red')  
    test_green_dir = os.path.join(test_dir, 'green')  
    test_orange_dir = os.path.join(test_dir, 'orange')  
    test_blue_dir = os.path.join(test_dir, 'blue')  
      
    # 훈련용 이미지 파일 이름 조회  
    train_white_fnames = os.listdir(train_white_dir)  
    train_red_fnames = os.listdir(train_red_dir)  
    train_green_fnames = os.listdir(train_green_dir)  
    train_orange_fnames = os.listdir(train_orange_dir)  
    train_blue_fnames = os.listdir(train_blue_dir)  
    print(train_white_fnames)  
    print(train_red_fnames)  
    print(train_green_fnames)  
    print(train_orange_fnames)  
    print(train_blue_fnames)  
</code>  

# Shape

# String  
![다운로드 (3)](https://user-images.githubusercontent.com/77331459/194784410-d8690c98-46e6-429f-8125-36897550d5d6.png)  

OCR은 Optical Character Recognition의 약자로 사람이 쓰거나 기계로 인쇄한 문자의 영상을 이미지 스캐너로 획득하여 기계가 읽을 수 있는 문자로 변환하는 것을 뜻한다.  
파이썬에서 사용할 라이브러리는 pytesseract이다.  
<code>
테서랙트(Tesseract)는 다양한 운영 체제를 위한 광학 문자 인식 엔진이다. 이 소프트웨어는 Apache License, 버전 2.0,에 따라 배포되는 무료 소프트웨어이며 2006년부터 Google에서 개발을 후원했다.
</code>  

기능

* get_languages Tesseract OCR에서 현재 지원하는 모든 언어를 반환합니다. 
* get_tesseract_version 시스템에 설치된 Tesseract 버전을 반환합니다.
* image_to_string Tesseract OCR 처리에서 수정되지 않은 출력을 문자열로 반환합니다.
* image_to_boxes 인식 된 문자와 해당 상자 경계를 포함하는 결과를 반환합니다.
* image_to_data 상자 경계, 신뢰도 및 기타 정보가 포함 된 결과를 반환합니다. Tesseract 3.05 이상이 필요합니다. 자세한 내용은 Tesseract TSV 문서 를 확인하십시오.
* image_to_osd 방향 및 스크립트 감지에 대한 정보가 포함 된 결과를 반환합니다.
* image_to_alto_xml Tesseract의 ALTO XML 형식의 형식으로 결과를 반환합니다.
* run_and_get_output Tesseract OCR에서 원시 출력을 반환합니다. tesseract로 전송되는 매개 변수를 좀 더 제어 할 수 있습니다.

매개 변수

* image 객체 또는 문자열-Tesseract에서 처리 할 이미지의 PIL 이미지 / NumPy 배열 또는 파일 경로입니다. 파일 경로 대신 객체를 전달하면 pytesseract는 암시 적으로 이미지를 RGB 모드 로 변환 합니다 .
* lang String-Tesseract 언어 코드 문자열입니다. 지정되지 않은 경우 기본값은 eng입니다 ! 여러 언어의 예 : lang = 'eng + fra'
* config String- pytesseract 함수를 통해 사용할 수없는 추가 사용자 지정 구성 플래그 입니다. 예 : config = '-psm 6'
* nice Integer-Tesseract 실행에 대한 프로세서 우선 순위를 수정합니다. Windows에서는 지원되지 않습니다. Nice는 유닉스와 유사한 프로세스의 우수성을 조정합니다.
* output_type 클래스 속성-출력 유형을 지정하며 기본값은 string 입니다. 지원되는 모든 유형의 전체 목록은 pytesseract.Output 클래스 의 정의를 확인하세요 .
* timeout Integer 또는 Float-OCR 처리를위한 기간 (초). 그 후 pytesseract가 종료되고 RuntimeError가 발생합니다.
* pandas_config Dict- Output.DATAFRAME 유형 에만 해당됩니다 . pandas.read_csv에 대한 사용자 지정 인수가있는 사전 . image_to_data 의 출력을 사용자 정의 할 수 있습니다 .



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
> 커밋 제목은 최대 50자 입력  

본문은 한 줄 최대 72자 입력  

Commit 메세지  


🪛[chore]: 코드 수정, 내부 파일 수정.  

✨[feat]: 새로운 기능 구현.  

🎨[style]: 스타일 관련 기능.(코드의 구조/형태 개선)  

➕[add]: Feat 이외의 부수적인 코드 추가, 라이브러리 추가  

🔧[file]: 새로운 파일 생성, 삭제 시  

🐛[fix]: 버그, 오류 해결.  

🔥[del]: 쓸모없는 코드/파일 삭제.  

📝[docs]: README나 WIKI 등의 문서 개정.  

💄[mod]: storyboard 파일,UI 수정한 경우.  

✏️[correct]: 주로 문법의 오류나 타입의 변경, 이름 변경 등에 사용합니다.  

🚚[move]: 프로젝트 내 파일이나 코드(리소스)의 이동  

⏪️[rename]: 파일 이름 변경이 있을 때 사용합니다.  

⚡️[improve]: 향상이 있을 때 사용합니다.  

♻️[refactor]: 전면 수정이 있을 때 사용합니다.  

🔀[merge]: 다른브렌치를 merge 할 때 사용합니다.  

✅ [test]: 테스트 코드를 작성할 때 사용합니다.  
