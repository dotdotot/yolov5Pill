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

</code>  
  
이미지 파일 이름 조회  
os.listdir()을 사용하여 경로 내에 있는 파일의 이름을 리스트의 형태로 반환받아 확인합니다.  
  
<code>  
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
      
    #각 디렉토리별 이미지 개수 확인  
      
    print('Total training red images :', len(os.listdir(train_red_dir)))  
    print('Total training green images :', len(os.listdir(train_green_dir)))  
    print('Total training blue images :', len(os.listdir(train_blue_dir)))  
    print('Total training orange images :', len(os.listdir(train_orange_dir)))  
    print('Total training white images :', len(os.listdir(train_white_dir)))  
      
    print('Total validation white images :', len(os.listdir(validation_white_dir)))  
    print('Total validation red images :', len(os.listdir(validation_red_dir)))  
    print('Total validation green images :', len(os.listdir(validation_green_dir)))  
    print('Total validation orange images :', len(os.listdir(validation_orange_dir)))  
    print('Total validation blue images :', len(os.listdir(validation_blue_dir)))  
      
    print('Total test white images :', len(os.listdir(test_white_dir)))  
    print('Total test red images :', len(os.listdir(test_red_dir)))  
    print('Total test green images :', len(os.listdir(test_green_dir)))  
    print('Total test orange images :', len(os.listdir(test_orange_dir)))  
    print('Total test blue images :', len(os.listdir(test_blue_dir)))  

</code>  

![제목 없음](https://user-images.githubusercontent.com/77331459/194784686-c3704c6c-f58c-44ba-87ad-71e0fa3f3d9a.png)  
  
  
  
이미지 확인  

<code>
    import matplotlib.pyplot as plt  

    import matplotlib.image as mpimg  
      
    nrows, ncols = 4, 4  
    pic_index = 0  
      
    fig = plt.gcf()  
    fig.set_size_inches(ncols*3, nrows*3)  
      
    pic_index += 8  
      
    next_red_pix = [os.path.join(train_red_dir, fname) for fname in train_red_fnames[pic_index-8:pic_index]]  
    next_green_pix = [os.path.join(train_green_dir, fname) for fname in train_green_fnames[pic_index-8:pic_index]]  
    next_blue_pix = [os.path.join(train_blue_dir, fname) for fname in train_blue_fnames[pic_index-8:pic_index]]  
    next_orange_pix = [os.path.join(train_orange_dir, fname) for fname in train_orange_fnames[pic_index-8:pic_index]]  
    next_white_pix = [os.path.join(train_white_dir, fname) for fname in train_white_fnames[pic_index-8:pic_index]]  
      
    for i, img_path in enumerate(next_red_pix + next_green_pix + next_blue_pix + next_orange_pix + next_white_pix):  
        sp = plt.subplot(nrows, ncols, i + 1)  
        sp.axis('OFF')  
          
        img = mpimg.imread(img_path)  
        plt.imshow(img)  
      
    plt.show()  

</code>  

![제목 없음1](https://user-images.githubusercontent.com/77331459/194784768-71ddcf50-c429-48e0-99b5-d4625466bf2d.png)  
  

이미지 데이터 전처리  

데이터가 부족하다고 생각했습니다. 

적은 수의 이미지에서 모델이 최대한 많은 정보를 뽑아내서 학습할 수 있도록, augmentation을 적용하였습니다.  

Augmentation이라는 것은, 이미지를 사용할 때마다 임의로 변형을 가함으로써 마치 훨씬 더 많은 이미지를 보고 공부하는 것과 같은 학습 효과를 내게 해줍니다.  

기존의 데이터의 정보량을 보존한 상태로 노이즈를 주는 방식인데, 이는 다시 말하면, 내가 가지고 있는 정보량은 변하지 않고 단지 정보량에 약간의 변화를 주는 것으로, 딥러닝으로 분석된 데이터의 강하게 표현되는 고유의 특징을 조금 느슨하게 만들어는 것이라고 생각하면 됩니다.   

Augmentation을 통해 결과적으로 과적합(오버피팅)을 막아 모델이 학습 데이터에만 맞춰지는 것을 방지하고, 새로운 이미지도 잘 분류할 수 있게 만들어 예측 범위도 넓혀줄 수 있습니다.  

이런 전처리 과정을 돕기 위해 케라스는 ImageDataGenerator 클래스를 제공합니다. ImageDataGenerator는 아래와 같은 일을 할 수 있습니다  

* 학습 과정에서 이미지에 임의 변형 및 정규화 적용  
* 변형된 이미지를 배치 단위로 불러올 수 있는 generator 생성  
- generator를 생성할 때 flow(data, labels), flow_from_directory(directory) 두 가지 함수를 사용 할 수 있습니다.  
- fit_generator(fit), evaluate_generator 함수를 사용하여 generator로 이미지를 불러와 모델을 학습시키고 평가 할 수 있습니다.  
  
이미지 데이터 생성  

ImageDataGenerator를 통해서 데이터를 만들어줄 것입니다.   

어떤 방식으로 데이터를 증식시킬 것인지 아래와 같은 옵션을 통해서 설정합니다.    

참고로, augmentation은 train 데이터에만 적용시켜야 하고, validation 및 test 이미지는 augmentation을 적용하지 않습니다.  

모델 성능을 평가할 때에는 이미지 원본을 사용해야 하기에 rescale만 적용해 정규화하고 진행합니다  

<code>
# 이미지 데이터 전처리
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
    #train셋에만 적용
    train_datagen = ImageDataGenerator(rescale = 1./255, # 모든 이미지 원소값들을 255로 나누기  
                                    rotation_range=25, # 0~25도 사이에서 임의의 각도로 원본이미지를 회전  
                                    width_shift_range=0.05, # 0.05범위 내에서 임의의 값만큼 임의의 방향으로 좌우 이동  
                                    height_shift_range=0.05, # 0.05범위 내에서 임의의 값만큼 임의의 방향으로 상하 이동  
                                    zoom_range=0.2, # (1-0.2)~(1+0.2) => 0.8~1.2 사이에서 임의의 수치만큼 확대/축소  
                                    horizontal_flip=True, # 좌우로 뒤집기                                     
                                    vertical_flip=True,  
                                    fill_mode='nearest'  
                                    )   
    #validation 및 test 이미지는 augmentation을 적용하지 않는다;  
    #모델 성능을 평가할 때에는 이미지 원본을 사용 (rescale만 진행)  
    validation_datagen = ImageDataGenerator(rescale = 1./255)  
    test_datagen = ImageDataGenerator(rescale = 1./255)   

</code>  

이미지 데이터 수가 적어서, batch_size를 결정하는 것에 여러 시행착오와 어려움이 있을것이라고 생각했습니다.  

Generator 생성시 batch_size와 steps_per_epoch(model fit할 때)를 곱한 값이 훈련 샘플 수 보다 작거나 같아야 합니다.   

이에 맞춰, flow_from_directory() 옵션에서 batch_size와 model fit()/fit_generator() 옵션의 steps_per_epoch 값을 조정해 가며 학습을 시도하였습니다.  

<code>
    #flow_from_directory() 메서드를 이용해서 훈련과 테스트에 사용될 이미지 데이터를 만들기
    #변환된 이미지 데이터 생성
    train_generator = train_datagen.flow_from_directory(train_dir,   
                                                        batch_size=16, # 한번에   변환된 이미지 16개씩   만들어라 라는 것  
                                                        color_mode='rgba', # 흑백   이미지 처리  
                                                        class_mode='categorical',   
                                                        target_size=(150,150)) #   target_size에 맞춰서   이미지의 크기가 조절된다  
    validation_generator = validation_datagen.flow_from_directory(validation_dir,   
                                                                batch_size=4,   
                                                                color_mode='rgba',  
                                                                class_mode='categorical',   
                                                                target_size=(150,  150))  
    test_generator = test_datagen.flow_from_directory(test_dir,  
                                                    batch_size=4,  
                                                    color_mode='rgba',  
                                                    class_mode='categorical',  
                                                    target_size=(150,150))  
    #참고로, generator 생성시 batch_size x steps_per_epoch (model fit에서) <= 훈련 샘플 수 보다 작거나 같아야 한다.  

</code>  
  
<code>
    # class 확인  
    train_generator.class_indices  
</code>
  
모델 구성  

합성곱 신경망 모델을 구성합니다.  

![image](https://user-images.githubusercontent.com/77331459/194784951-18705042-9f54-43c5-9982-4844afe8e629.png)  
  
모델 학습  

모델 컴파일 단계에서는 compile() 메서드를 이용해서 손실 함수(loss function)와 옵티마이저(optimizer)를 지정합니다.  

* 손실 함수로 ‘binary_crossentropy’를 사용했습니다.(변경 예정)
* 또한, 옵티마이저로는 RMSprop을 사용했습니다. RMSprop(Root Mean Square Propagation) 알고리즘은 훈련 과정 중에 학습률을 적절하게 변화시켜 줍니다.  
* 훈련과 테스트를 위한 데이터셋인 train_generator, validation_generator를 입력합니다.
* epochs는 데이터셋을 한 번 훈련하는 과정을 의미합니다.
* steps_per_epoch는 한 번의 에포크 (epoch)에서 훈련에 사용할 배치 (batch)의 개수를 지정합니다.
* validation_steps는 한 번의 에포크가 끝날 때, 테스트에 사용되는 배치 (batch)의 개수를 지정합니다.

<code>  
    from tensorflow.keras.optimizers import RMSprop  
  
    #compile() 메서드를 이용해서 손실 함수 (loss function)와 옵티마이저 (optimizer)를 지정  
    model.compile(optimizer=RMSprop(learning_rate=0.001), # 옵티마이저로는 RMSprop 사용  
                loss='binary_crossentropy', # 손실 함수로 ‘sparse_categorical_crossentropy’ 사용  
                metrics= ['accuracy'])  
    # RMSprop (Root Mean Square Propagation) Algorithm: 훈련 과정 중에 학습률을 적절하게 변화시킨다.  
  
  
  
    #모델 훈련  
    history = model.fit_generator(train_generator, # train_generator안에 X값, y값 다 있으니 generator만 주면 된다  
                                validation_data=validation_generator, # validatino_generator안에도 검증용 X,y데이터들이 다 있으니 generator로 주면 됨  
                                steps_per_epoch=4, # 한 번의 에포크(epoch)에서 훈련에 사용할 배치(batch)의 개수 지정; generator를 4번 부르겠다  
                                epochs=100, # 데이터셋을 한 번 훈련하는 과정; epoch은 100 이상은 줘야한다  
                                validation_steps=4, # 한 번의 에포크가 끝날 때, 검증에 사용되는 배치(batch)의 개수를 지정; validation_generator를 4번 불러서 나온 이미지들로 작업을 해라  
                                verbose=2)  
    #참고: validation_steps는 보통 내가 원하는 이미지 수에 flow할 때 지정한 batchsize로 나눈 값을 validation_steps로 지정  

</code>
  
  
결과 확인 및 평가  
학습된 모델 결과와 성능을 확인합니다.  

<code>

    # 모델 성능 평가  
    model.evaluate(train_generator)  
</code>  

<code>

    model.evaluate(validation_generator)        
</code>  
  
    
정확도 및 손실 시각화  

훈련 과정에서 epoch에 따른 정확도와 손실을 시각화화여 확인합니다.  

<code>

    # 정확도 및 손실 시각화  
    acc = history.history['accuracy']  
    val_acc = history.history['val_accuracy']  
    loss = history.history['loss']  
    val_loss = history.history['val_loss']  
      
    epochs = range(len(acc))  
      
    plt.plot(epochs, acc, 'bo', label='Training accuracy')  
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')  
    plt.title('Training and validation accuracy')  
    plt.legend()  
      
    plt.figure()  
      
    plt.plot(epochs, loss, 'go', label='Training loss')  
    plt.plot(epochs, val_loss, 'g', label='Validation loss')  
    plt.title('Training and validation loss')  
    plt.legend()  
      
    plt.show()  
    
</code>

![image (1)](https://user-images.githubusercontent.com/77331459/194785124-827a1941-125f-4912-b36a-17be341d4d54.png)  
  
테스트 평가  
#

- datagenImage1 

- epoch 50 , patience=5
batchsize 4
iterations 5

테스트용
<code>

    ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2)
    testImage folder = vsCode\\PillProject\\imageT\\color\\test
    target_size = 256,256
    batchsize = 4
    class_mode = categorical
</code>

훈련용(설정은 위와 동일)</br>
trainImage folder = vsCode\\PillProject\\imageT\\color\\train</br>

batch_size = 64</br>
num_classes = 5</br>
epochs = 50</br>


<img width="373" alt="datagenImage1학습이미지(patience=5)" src="https://user-images.githubusercontent.com/77331459/205564571-cd6c7f15-97ae-4b2a-bdf4-6480d578ab82.png">

신경망 구성

<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )<
</code>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        callbacks=[early_stopping]
    )
</code>

18에서 학습중지 (최대 50)</br>
Loss : 0.7770828008651733, Acc : 0.8653846383094788</br>

<img width="251" alt="datagenImage1결과(patience=5)" src="https://user-images.githubusercontent.com/77331459/205564584-c17a2b86-9722-4c62-953a-81a6a01105e8.png">

<img width="454" alt="datagenImage1검증(patience=5)" src="https://user-images.githubusercontent.com/77331459/205564559-34f99053-f66f-443a-a9f5-1c9966a490b2.png">

#

- datagenImage1 

- epoch 50 , patience = 10
batchsize 4
iterations 5

테스트용
<code>

    ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2)
    testImage folder = vsCode\\PillProject\\imageT\\color\\test
    target_size = 256,256
    batchsize = 4
    class_mode = categorical
</code>

훈련용(설정은 위와 동일)</br>
trainImage folder = vsCode\\PillProject\\imageT\\color\\train</br>

batch_size = 64</br>
num_classes = 5</br>
epochs = 50</br>

<img width="370" alt="datagenImage1학습이미지(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564671-0d98e75f-9e3c-4e3b-b054-6f1b21ad5345.png">

신경망 구성

<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )<
</code>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        callbacks=[early_stopping]
    )
</code>

30에서 학습중지 (최대 50)</br>
Loss : 2.2101550102233887, Acc : 0.699999988079071</br>

<img width="248" alt="datagenImage1결과(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564680-ce4f6eaa-4a2a-4025-b0af-ccfc39b8e297.png">

<img width="465" alt="datagenImage1검증(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564686-247119cc-75ac-4ded-a589-ed70fddf95f7.png">

#

- datagenImage1 

- epoch 50 , patience = 20</br>
batchsize 4 </br>
iterations 5 </br>

테스트용
<code>

    ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2)
    testImage folder = vsCode\\PillProject\\imageT\\color\\test
    target_size = 256,256
    batchsize = 4
    class_mode = categorical
</code>

훈련용(설정은 위와 동일)</br>
trainImage folder = vsCode\\PillProject\\imageT\\color\\train</br>

batch_size = 64</br>
num_classes = 5</br>
epochs = 50</br>

<img width="374" alt="datagenImage1학습이미지(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564637-c238ab2e-3059-48a1-8157-f7089819c56d.png">

신경망 구성

<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )<
</code>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        callbacks=[early_stopping]
    )
</code>

50까지 학습 완료 (최대 50)</br>
Loss : 0.7318955659866333, Acc : 1.0</br>

<img width="259" alt="datagenImage1결과(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564626-ada0b971-d632-4b77-acfe-53f874463ee4.png">

<img width="460" alt="datagenImage1검증(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564646-fbd46928-91fa-4499-a46d-ccf3cc10869c.png">

#

- datagenImage2

- epoch 50 , patience = 10</br>
batchsize 4 </br>

훈련용
<code>

    datagen = ImageDataGenerator(
    featurewise_center = True)

    datagen.flow_from_directory(
    'C:\\vsCode\\PillProject\\imageT\\color\\test', 
    shuffle = True, 
    target_size=(256,256), 
    batch_size=batch_size, 
    class_mode = 'categorical')
</code>

테스트용(경로 제외 위와 동일)</br>
C:\\vsCode\\PillProject\\imageT\\color\\train</br>

num_classes = 5</br>
epochs = 50</br>

<img width="369" alt="datagenImage2학습이미지(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564721-7181cb4f-7bc7-4973-b93c-859fb2358b7c.png">

신경망 구성

<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )<
</code>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=30)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        callbacks=[early_stopping]
    )
</code>

35까지 학습 완료 (최대 50)</br>
Loss : 0.023338522762060165, Acc : 1.0</br>

<img width="261" alt="datagenImage2결과(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564730-979169c2-b521-4756-8a78-cd2607ae81f6.png">

<img width="453" alt="datagenImage2검증(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564736-e0b6c974-7fa4-43c3-8844-be7b84df4239.png">

#

- datagenImage2

- epoch 50 , patience = 20</br>
batchsize 4 </br>

훈련용
<code>

    datagen = ImageDataGenerator(
    featurewise_center = True)

    datagen.flow_from_directory(
    'C:\\vsCode\\PillProject\\imageT\\color\\test', 
    shuffle = True, 
    target_size=(256,256), 
    batch_size=batch_size, 
    class_mode = 'categorical')
</code>

테스트용(경로 제외 위와 동일)</br>
C:\\vsCode\\PillProject\\imageT\\color\\train</br>

num_classes = 5</br>
epochs = 50</br>

<img width="367" alt="datagenImage2학습이미지(patience = 20)" src="https://user-images.githubusercontent.com/77331459/205564745-cd704b5f-ca7a-49de-a24c-020d6dc42212.png">

신경망 구성

<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )<
</code>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=30)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        callbacks=[early_stopping]
    )
</code>

35까지 학습 완료 (최대 50)</br>
Loss : 0.38273733854293823, Acc : 0.921875</br>

<img width="270" alt="datagenImage2결과(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564753-0a10c105-c897-466a-9205-afb2cf4d8f00.png">

<img width="441" alt="datagenImage2검증(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564767-ff55c2d3-d4a8-40ea-bdc1-892d98150114.png">


#

- datagenImage2

- epoch 50 , patience = 30</br>
batchsize 4 </br>

훈련용
<code>

    datagen = ImageDataGenerator(
    featurewise_center = True)

    datagen.flow_from_directory(
    'C:\\vsCode\\PillProject\\imageT\\color\\test', 
    shuffle = True, 
    target_size=(256,256), 
    batch_size=batch_size, 
    class_mode = 'categorical')
</code>

테스트용(경로 제외 위와 동일)</br>
C:\\vsCode\\PillProject\\imageT\\color\\train</br>

num_classes = 5</br>
epochs = 50</br>

<img width="370" alt="datagenImage2학습이미지(patience = 30)" src="https://user-images.githubusercontent.com/77331459/205564801-139e2d6a-954c-4bb2-89bd-6bc7a8867203.png">

신경망 구성

<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )<
</code>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=30)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        callbacks=[early_stopping]
    )
</code>

50까지 학습 완료 (최대 50)</br>
Loss : 0.10279396176338196, Acc : 0.953125</br>

<img width="267" alt="datagenImage2결과(patience=30)" src="https://user-images.githubusercontent.com/77331459/205564843-c8d2d900-d0bf-4a5a-a9f3-a753319f9584.png">

<img width="468" alt="datagenImage2검증(patience=30)" src="https://user-images.githubusercontent.com/77331459/205564865-b575dfc7-3d21-4941-8d24-88d37f21bdca.png">


# Shape  
color와 동일한 cnn모델을 사용하였음

테스트 평가
#

epoch 5000  |  patience 100</br>

batch_size = 4</br>
#생성시에 파라미터를 설정하면 어떻게 augmentation를 진행할지 지정할 수 있다.</br>
<code>

    datagen = ImageDataGenerator(
        featurewise_center = True)
</code>

테스트용</br>

<code>

    #경로, 셔플, 이미지사이즈, 한번에 읽어올 이미지 수, 클래스 모드
    generator = datagen.flow_from_directory(
        'C:\\vsCode\\PillProject\\imageT\\shape\\test', 
        shuffle = True, 
        target_size=(256,256), 
        batch_size=batch_size, 
        class_mode = 'categorical',
        color_mode='grayscale')
    훈련용(경로만 다름)
    C:\\vsCode\\PillProject\\imageT\\shape\\train

    class_names = ['circle', 'hexagon', 'pentagon', 'rectangle', 'rectangular', 'triangle']
</code>

<img width="366" alt="shapeDatagenImage1학습이미지(patience = 100)" src="https://user-images.githubusercontent.com/77331459/205568304-b21fb1b0-7542-44fe-ad87-db7da1fb1fd9.png">

#배치 사이즈의 수만큼 이미지를 학습하고 가중치를 갱신하게된다.</br>
#배치 사이즈를 증가시키면 필요한 메모리가 증가하나 모델을 훈련하는데 시간이 적게 든다.</br>
#배치 사이즈를 감소시키면 필요한 메모리가 감소하나 모델을 훈련하는데 시간이 많이 든다.</br>
batch_size = 64</br>
#분류될 클래스 개수</br>
num_classes = 6</br>
#몇번 학습을 반복할 것인지 결정</br>
#에포크가 많다면 과적합 문제 발생가능, 적다면 분류를 제대로 못할 수 있다.</br>
epochs = 5000</br>

#모델 구성
<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

#모델을 학습시키기 전 환경 설정 (정규화기, 손실함수, 평가지표)</br>

#정규화기 - 훈련과정을 설정합니다. 즉, 최적화 알고리즘을 설정을 의미합니다.</br>
#adam, sgd, rmsprop, adagrad 등이 있습니다.</br>
#분류에는  ‘SGD’, ‘Adam’, ‘RMSprop’</br>

#손실함수 - 모델이 최적화에 사용되는 목적 함수입니다.</br>
#mse, categorical_crossentropy, binary_crossentropy 등이 있습니다.</br>

#평가지표 - 훈련을 모니터링 하기 위해 사용됩니다.</br>
#분류에서는 accuracy, 회귀에서는 mse, rmse, r2, mae, mspe, mape, msle 등이 있습니다.</br>
#사용자가 메트릭을 정의해서 사용할 수도 있습니다.</br>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )
</code>

#과적합을 방지하기 위해서 설정</br>
#Early Stopping 이란 너무 많은 Epoch 은 overfitting 을 일으킨다. 하지만 너무 적은 Epoch 은 underfitting 을 일으킨다. </br>
#라는 딜레마를 해결하기위함</br>

#Epoch 을 정하는데 많이 사용되는 Early stopping 은 무조건 Epoch 을 많이 돌린 후, 특정 시점에서 멈추는 것이다. </br>
#그 특정시점을 어떻게 정하느냐가 Early stopping 의 핵심이라고 할 수 있다. </br>
#일반적으로 hold-out validation set 에서의 성능이 더이상 증가하지 않을 때 학습을 중지</br>
#https://3months.tistory.com/424 참고</br>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=100)
    #ModelCheckpoint instance를 callbacks 파라미터에 넣어줌으로써, 가장 validation performance 가 좋았던 모델을 저장할 수 있게된다.
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        # 콜백을 적용하면 과적합을 방지하는지는 잘 모르겠음.
        # 다만, 적용시키면 훈련을 12~20번 사이로 하고 그냥 에포크 상관없이 훈련을 중지함.
        callbacks=[early_stopping,mc]
    )
</code>

119 까지만 학습 (최대 5000)</br>
Loss : 4.203685283660889, Acc : 0.5714285969734192</br>

<img width="267" alt="shapeDatagenImage1결과(patience = 100)" src="https://user-images.githubusercontent.com/77331459/205568319-e5876d66-b066-4396-a993-10d3c7638135.png">

<img width="476" alt="shapeDatagenImage1검증(patience = 100)" src="https://user-images.githubusercontent.com/77331459/205568331-fe73dc87-0817-44b4-bd22-46755b7d734c.png">

#

epoch 5000  |  patience 50</br>

batch_size = 4</br>
#생성시에 파라미터를 설정하면 어떻게 augmentation를 진행할지 지정할 수 있다.</br>
<code>

    datagen = ImageDataGenerator(
        featurewise_center = True)
</code>

테스트용</br>

<code>

    #경로, 셔플, 이미지사이즈, 한번에 읽어올 이미지 수, 클래스 모드
    generator = datagen.flow_from_directory(
        'C:\\vsCode\\PillProject\\imageT\\shape\\test', 
        shuffle = True, 
        target_size=(256,256), 
        batch_size=batch_size, 
        class_mode = 'categorical',
        color_mode='grayscale')
    훈련용(경로만 다름)
    C:\\vsCode\\PillProject\\imageT\\shape\\train

    class_names = ['circle', 'hexagon', 'pentagon', 'rectangle', 'rectangular', 'triangle']
</code>

<img width="379" alt="shapeDatagenImage1학습이미지(patience = 50)" src="https://user-images.githubusercontent.com/77331459/205567847-482f8737-d5fa-4b1c-ab45-1ce391dc1b9e.png">

#배치 사이즈의 수만큼 이미지를 학습하고 가중치를 갱신하게된다.</br>
#배치 사이즈를 증가시키면 필요한 메모리가 증가하나 모델을 훈련하는데 시간이 적게 든다.</br>
#배치 사이즈를 감소시키면 필요한 메모리가 감소하나 모델을 훈련하는데 시간이 많이 든다.</br>
batch_size = 64</br>
#분류될 클래스 개수</br>
num_classes = 6</br>
#몇번 학습을 반복할 것인지 결정</br>
#에포크가 많다면 과적합 문제 발생가능, 적다면 분류를 제대로 못할 수 있다.</br>
epochs = 5000</br>

#모델 구성
<code>

    model = keras.Sequential([
        Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = train_images.shape[1:],
            activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Conv2D(64, kernel_size = (3,3), padding = 'same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])
</code>

#모델을 학습시키기 전 환경 설정 (정규화기, 손실함수, 평가지표)</br>

#정규화기 - 훈련과정을 설정합니다. 즉, 최적화 알고리즘을 설정을 의미합니다.</br>
#adam, sgd, rmsprop, adagrad 등이 있습니다.</br>
#분류에는  ‘SGD’, ‘Adam’, ‘RMSprop’</br>

#손실함수 - 모델이 최적화에 사용되는 목적 함수입니다.</br>
#mse, categorical_crossentropy, binary_crossentropy 등이 있습니다.</br>

#평가지표 - 훈련을 모니터링 하기 위해 사용됩니다.</br>
#분류에서는 accuracy, 회귀에서는 mse, rmse, r2, mae, mspe, mape, msle 등이 있습니다.</br>
#사용자가 메트릭을 정의해서 사용할 수도 있습니다.</br>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )
</code>

#과적합을 방지하기 위해서 설정</br>
#Early Stopping 이란 너무 많은 Epoch 은 overfitting 을 일으킨다. 하지만 너무 적은 Epoch 은 underfitting 을 일으킨다. </br>
#라는 딜레마를 해결하기위함</br>

#Epoch 을 정하는데 많이 사용되는 Early stopping 은 무조건 Epoch 을 많이 돌린 후, 특정 시점에서 멈추는 것이다. </br>
#그 특정시점을 어떻게 정하느냐가 Early stopping 의 핵심이라고 할 수 있다. </br>
#일반적으로 hold-out validation set 에서의 성능이 더이상 증가하지 않을 때 학습을 중지</br>
#https://3months.tistory.com/424 참고</br>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=50)
    #ModelCheckpoint instance를 callbacks 파라미터에 넣어줌으로써, 가장 validation performance 가 좋았던 모델을 저장할 수 있게된다.
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        # 콜백을 적용하면 과적합을 방지하는지는 잘 모르겠음.
        # 다만, 적용시키면 훈련을 12~20번 사이로 하고 그냥 에포크 상관없이 훈련을 중지함.
        callbacks=[early_stopping,mc]
    )
</code>

119 까지만 학습 (최대 5000)</br>
Loss : 4.203685283660889, Acc : 0.5714285969734192</br>

<img width="275" alt="shapeDatagenImage1결과(patience = 50)" src="https://user-images.githubusercontent.com/77331459/205567857-b39c44eb-e9e5-4aeb-8722-e19768207f47.png">

<img width="461" alt="shapeDatagenImage1검증(patience = 50)" src="https://user-images.githubusercontent.com/77331459/205567868-e0f25d31-cce9-4db9-8383-b6472402189b.png">

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
