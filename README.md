# Pill Project
๐ ํ๋ก์ ํธ ์๊ฐ : Pill Project</br>
๐ ๊ธฐ๊ฐ : 2022.9.24 ~   </br>
๐ ํ์:  [์ค์](https://github.com/dotdotot)</br>
๐ ๋ฆฌ๋ทฐ์ด: [์ค์](https://github.com/dotdotot)</br></br>

# Yolov5
yolo ์ปค์คํ ๋ฐ์ดํฐ์ ํ์ต์ํค๊ธฐ(colab)</br>

1. ๋๋ผ์ด๋ธ ๋ง์ดํธ</br>
<code>
    #๋๋ผ์ด๋ธ ๋ง์ดํธ</br>

    from google.colab import drive

    drive.mount('/content/drive')
</code>

2. ํ๊ฒฝ ์ธํ</br>
<code>
    ๋ด ๊ตฌ๊ธ ๋๋ผ์ด๋ธ๋ก ์ด๋

    %cd "/content/drive/MyDrive"

    Yolov5 github ๋ ํฌ์งํ ๋ฆฌ clone

    !git clone https://github.com/ultralytics/yolov5.git

    ํ์ํ ๋ชจ๋ ์ค์น
    !pip install -U -r yolov5/requirements.txt
</code>
<code>
    import torch

    #ํ์ดํ ์น ๋ฒ์  ํ์ธ, cuda device properties ํ์ธ

    print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
<code>

๋ฐํ์ -> ๋ฐํ์ ์ ํ ๋ณ๊ฒฝ์ ๋ค์ด๊ฐ์ ํ๋์จ์ด ๊ฐ์๊ธฐ GPU๋ก ๋ณ๊ฒฝ</br>
์ดํ custom ๋ชจ๋ธ์ ์ฌ์ฉํ  ๋ฐ์ดํฐ ์์งํ๊ธฐ</br>

3. data.yaml ํ์ผ ์์ฑ</br>
data.yaml : ๋ชจ๋ธ ํ์ต์ ์ํ ์์ ํํ ๋ฆฌ์ผ ๊ฐ์ ๊ฒ (๋ด ๋๋ผ์ด๋ธ/dataset ์๋์ ๋ง๋ค์ด์ค๋ค)  

![๋ค์ด๋ก๋](https://user-images.githubusercontent.com/77331459/194784144-00d6d2a6-9074-4eed-9f9b-6cca9decdd79.png)  

ํ์ผ์ ๋ด์ฉ์ ๊ทธ๋ฆผ๊ณผ ๊ฐ์ด ์ ์ด์ฃผ๋ฉด ๋๋ค.  

class๊ฐ ์ฌ๋ฌ๊ฐ๋ผ๋ฉด nc์ ๊ฐ์๋ฅผ class์ ๊ฐ์๋งํผ ์ง์ ํ๊ณ  names ๋ฐฐ์ด ๋ด๋ถ์ class ์ด๋ฆ์ ์ ์ด์ฃผ๋ฉด ๋๋ค.  

์ฆ, nc๋ ์์ ์ด ํ์ต์ํค๊ณ ์ ํ๋ ํด๋์ค์ ์(number)๊ณ   

names์๋ ๊ทธ ํด๋์ค์  ์ด๋ฆ์ ๋ฐฐ์ด๋ก ์ ์ด์ฃผ๋ฉด ๋จ  

์ฌ๋ฌ๊ฐ์ผ ๊ฒฝ์ฐ ['class1','class2'] ์ฒ๋ผ..    



4.  labels/images ํด๋ ์ ๋ฆฌํด์ฃผ๊ธฐ  

2๋ฒ์์ ๋ง๋ค์ด๋จ๋ ๋ด ๋ฐ์ดํฐ์๋ค์ ๋ชจ๋ ์ ๋ฆฌํด์ค์ผํ๋ค.  

images/train ์๋ ํ๋ จ์ํค๊ณ ์ ํ๋ image๋ค์ ๋ฃ๊ณ   

images/val ์๋ validation์ ์ฌ์ฉ๋๋ image๋ค,   

labels/train  ํ๋ จ์ํค๋ image์ ๋ฐ์ด๋ฉ ๋ฐ์ค ์ ๋ณด๊ฐ ์ ์ฅ๋ txtํ์ผ๋ค  

labels/val ์๋  validation์ ์ฌ์ฉ๋๋  image์ ๋ฐ์ด๋ฉ ๋ฐ์ค ์ ๋ณด txtํ์ผ๋ค ์ ๋ชจ๋ ์๋ก๋ ํด์ค๋ค.  


5. ๋ชจ๋ธ ์ ํํ๊ธฐ  

![๋ค์ด๋ก๋ (1)](https://user-images.githubusercontent.com/77331459/194784149-35ee09f9-91a9-42a0-917b-6b39c85f147d.png)   


yolov5/models ์ ์ฌ๋ฌ ํ์ผ๋ค์ด ์๋ค. ๊ทธ ์ค ํ๋ ์ ํํ์ฌ ํด๋น ํ์ผ ๋ด์ฉ์ค nc ๋ฅผ ์์ ์ด ํ์ต์ํค๊ณ ์ ํ๋ ํด๋์ค ๊ฐ์๋ก ๋ฐ๊พผ๋ค.  

![๋ค์ด๋ก๋ (2)](https://user-images.githubusercontent.com/77331459/194784150-8db0c7dc-515d-4467-8408-dd77a975670a.png)    



6. training ์ํค๊ธฐ !!  

training ์ํค๊ธฐ ์ ์ ํญ์ ์ด ์ฝ๋๋ฅผ ์คํํด์ค๋ค  

<code>
    !pip install -U PyYAML
</code>  

์ฝ๋๋ฅผ ํตํด yolov5 ๋๋ ํ ๋ฆฌ๋ก ์ด๋  

<code>
    %cd /content/drive/My\ Drive/yolov5
</code>  

img: ์๋ ฅ ์ด๋ฏธ์ง ํฌ๊ธฐ  

batch: ๋ฐฐ์น ํฌ๊ธฐ  

epochs: ํ์ต epoch ์ (์ฐธ๊ณ : 3000๊ฐ ์ด์์ด ์ผ๋ฐ์ ์ผ๋ก ์ฌ์ฉ๋๋ค๊ณ  ํ๋ค...)  

data: data.yaml ํ์ผ ๊ฒฝ๋ก  

cfg: ๋ชจ๋ธ ๊ตฌ์ฑ ์ง์   


weights: ๊ฐ์ค์น์ ๋ํ ์ฌ์ฉ์ ์ ์ ๊ฒฝ๋ก๋ฅผ ์ง์ ํฉ๋๋ค(์ฐธ๊ณ : Ultraalytics Google Drive ํด๋์์ ๊ฐ์ค์น๋ฅผ ๋ค์ด๋ก๋ํ  ์ ์์ต๋๋ค).  

name: ๋ชจ๋ธ์ด ์ ์ฅ ๋  ํด๋ ์ด๋ฆ  

nosave: ์ต์ข ์ฒดํฌํฌ์ธํธ๋ง ์ ์ฅ  

cache: ๋ ๋น ๋ฅธ ํ์ต์ ์ํด ์ด๋ฏธ์ง๋ฅผ ์บ์  
  


!python train.py --img 640 --batch 30 --epochs 100 --data /content/drive/My\ Drive/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name [๋ด๊ฐ ์ ํ ํด๋๋ช]  

๋ฅผ ์คํํ๋ฉด ํ๋ จ์ด ์์๋๋ค.  

ํ๋ จ์ด ์์๋๊ณ  ์๋ฃ๋๋ฉด ์ด๋ ํด๋์ .ptํ์ผ์ด ์์ฑ์ด ๋์๋์ง ํ์ธ๊ฐ๋ฅํ๋ค.  
  
  
7. yolov5 ์คํ  
cd /content/drive/MyDrive/yolov5 ์ฝ๋๋ฅผ ์คํํด .ptํ์ผ์ด ์กด์ฌํ๋ ์์น๋ก ์ด๋ํ๋ค  
#์ฌ์ฉ  

!python detect.py --img 640 --weights "/content/yolov5/runs/train/coustomYolov5m/weights/best.pt" --source "/content/drive/MyDrive/testImages"  



# Color  
CNN(Convolutional Neural Networks)  

CNN(Convolutional Neural Networks)์ ์๋์ผ๋ก ํน์ง์ ์ถ์ถํ  ํ์ ์์ด ๋ฐ์ดํฐ๋ก๋ถํฐ ์ง์  ํ์ตํ๋ ๋ฅ๋ฌ๋์ ์ํ ์ ๊ฒฝ๋ง ์ํคํ์ฒ  

ํน์ง ์ถ์ถ ์์ญ์ ํฉ์ฑ๊ณฑ์ธต(Convolution layer)๊ณผ ํ๋ง์ธต(Pooling layer)์ ์ฌ๋ฌ ๊ฒน ์๋ ํํ(Conv+Maxpool)๋ก ๊ตฌ์ฑ๋์ด ์์ผ๋ฉฐ, ์ด๋ฏธ์ง์ ํด๋์ค๋ฅผ ๋ถ๋ฅํ๋ ๋ถ๋ถ์ Fully connected(FC) ํ์ต ๋ฐฉ์์ผ๋ก ์ด๋ฏธ์ง๋ฅผ ๋ถ๋ฅ  

![images_eodud0582_post_00e763c2-8f36-44e9-9303-7a710256d8c9_image](https://user-images.githubusercontent.com/77331459/194784494-53df4a7a-f72a-498b-b6c7-cfe5d74ac9bf.png)
CNN ์๊ณ ๋ฆฌ์ฆ์ ๊ตฌ์กฐ  

  
CNN์ ์ฃผ๋ก ์ด๋ฏธ์ง๋ ์์ ๋ฐ์ดํฐ๋ฅผ ์ฒ๋ฆฌํ  ๋ ์ฐ์ด๋๋ฐ, ์์์์ ๊ฐ์ฒด, ์ผ๊ตด, ์ฅ๋ฉด ์ธ์์ ์ํ ํจํด์ ์ฐพ์ ๋ ํนํ ์ ์ฉํ๋ฉฐ, ์ค๋์ค, ์๊ณ์ด, ์ ํธ ๋ฐ์ดํฐ์ ๊ฐ์ด ์์ ์ด์ธ์ ๋ฐ์ดํฐ๋ฅผ ๋ถ๋ฅํ๋ ๋ฐ๋ ํจ๊ณผ์   

  
๊ณผ์  :  
1. ๋ฐ์ดํฐ์ ์ค๋น  
2. ๊ฒฝ๋ก ์ง์  ๋ฐ ๋ฐ์ดํฐ ์ดํด๋ณด๊ธฐ  
3. ์ด๋ฏธ์ง ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ  
4. ๋ชจ๋ธ ๊ตฌ์ฑ  
5. ๋ชจ๋ธ ํ์ต  
6. ํ์คํธ ํ๊ฐ  
7. ๋ชจ๋ธ ์ ์ฅ  
  
1. ๋ฐ์ดํฐ์ ์ค๋น  
train, validation, test ํด๋๋ฅผ ์์ฑ  
  
๊ฒฝ๋ก ์ง์   

<code>
    # ๊ธฐ๋ณธ ๊ฒฝ๋ก  
    base_dir = 'C:\\vsCode\PillProject\image\color\\'  
    train_dir = os.path.join(base_dir, 'train')  
    validation_dir = os.path.join(base_dir, 'validation')  
    test_dir = os.path.join(base_dir, 'test')  
      
    # ํ๋ จ์ฉ ์ด๋ฏธ์ง ๊ฒฝ๋ก  
    train_red_dir = os.path.join(train_dir, 'red')  
    train_green_dir = os.path.join(train_dir, 'green')  
    train_blue_dir = os.path.join(train_dir, 'blue')  
    train_orange_dir = os.path.join(train_dir, 'orange')  
    train_white_dir = os.path.join(train_dir, 'white')  
      
    # ๊ฒ์ฆ์ฉ ์ด๋ฏธ์ง ๊ฒฝ๋ก  
    validation_white_dir = os.path.join(validation_dir, 'white')  
    validation_red_dir = os.path.join(validation_dir, 'red')  
    validation_green_dir = os.path.join(validation_dir, 'green')  
    validation_orange_dir = os.path.join(validation_dir, 'orange')  
    validation_blue_dir = os.path.join(validation_dir, 'blue')  
      
    # ํ์คํธ์ฉ ์ด๋ฏธ์ง ๊ฒฝ๋ก  
    test_white_dir = os.path.join(test_dir, 'white')  
    test_red_dir = os.path.join(test_dir, 'red')  
    test_green_dir = os.path.join(test_dir, 'green')  
    test_orange_dir = os.path.join(test_dir, 'orange')  
    test_blue_dir = os.path.join(test_dir, 'blue')  

</code>  
  
์ด๋ฏธ์ง ํ์ผ ์ด๋ฆ ์กฐํ  
os.listdir()์ ์ฌ์ฉํ์ฌ ๊ฒฝ๋ก ๋ด์ ์๋ ํ์ผ์ ์ด๋ฆ์ ๋ฆฌ์คํธ์ ํํ๋ก ๋ฐํ๋ฐ์ ํ์ธํฉ๋๋ค.  
  
<code>  
    # ํ๋ จ์ฉ ์ด๋ฏธ์ง ํ์ผ ์ด๋ฆ ์กฐํ  
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
      
    #๊ฐ ๋๋ ํ ๋ฆฌ๋ณ ์ด๋ฏธ์ง ๊ฐ์ ํ์ธ  
      
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

![์ ๋ชฉ ์์](https://user-images.githubusercontent.com/77331459/194784686-c3704c6c-f58c-44ba-87ad-71e0fa3f3d9a.png)  
  
  
  
์ด๋ฏธ์ง ํ์ธ  

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

![์ ๋ชฉ ์์1](https://user-images.githubusercontent.com/77331459/194784768-71ddcf50-c429-48e0-99b5-d4625466bf2d.png)  
  

์ด๋ฏธ์ง ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ  

๋ฐ์ดํฐ๊ฐ ๋ถ์กฑํ๋ค๊ณ  ์๊ฐํ์ต๋๋ค. 

์ ์ ์์ ์ด๋ฏธ์ง์์ ๋ชจ๋ธ์ด ์ต๋ํ ๋ง์ ์ ๋ณด๋ฅผ ๋ฝ์๋ด์ ํ์ตํ  ์ ์๋๋ก, augmentation์ ์ ์ฉํ์์ต๋๋ค.  

Augmentation์ด๋ผ๋ ๊ฒ์, ์ด๋ฏธ์ง๋ฅผ ์ฌ์ฉํ  ๋๋ง๋ค ์์๋ก ๋ณํ์ ๊ฐํจ์ผ๋ก์จ ๋ง์น ํจ์ฌ ๋ ๋ง์ ์ด๋ฏธ์ง๋ฅผ ๋ณด๊ณ  ๊ณต๋ถํ๋ ๊ฒ๊ณผ ๊ฐ์ ํ์ต ํจ๊ณผ๋ฅผ ๋ด๊ฒ ํด์ค๋๋ค.  

๊ธฐ์กด์ ๋ฐ์ดํฐ์ ์ ๋ณด๋์ ๋ณด์กดํ ์ํ๋ก ๋ธ์ด์ฆ๋ฅผ ์ฃผ๋ ๋ฐฉ์์ธ๋ฐ, ์ด๋ ๋ค์ ๋งํ๋ฉด, ๋ด๊ฐ ๊ฐ์ง๊ณ  ์๋ ์ ๋ณด๋์ ๋ณํ์ง ์๊ณ  ๋จ์ง ์ ๋ณด๋์ ์ฝ๊ฐ์ ๋ณํ๋ฅผ ์ฃผ๋ ๊ฒ์ผ๋ก, ๋ฅ๋ฌ๋์ผ๋ก ๋ถ์๋ ๋ฐ์ดํฐ์ ๊ฐํ๊ฒ ํํ๋๋ ๊ณ ์ ์ ํน์ง์ ์กฐ๊ธ ๋์จํ๊ฒ ๋ง๋ค์ด๋ ๊ฒ์ด๋ผ๊ณ  ์๊ฐํ๋ฉด ๋ฉ๋๋ค.   

Augmentation์ ํตํด ๊ฒฐ๊ณผ์ ์ผ๋ก ๊ณผ์ ํฉ(์ค๋ฒํผํ)์ ๋ง์ ๋ชจ๋ธ์ด ํ์ต ๋ฐ์ดํฐ์๋ง ๋ง์ถฐ์ง๋ ๊ฒ์ ๋ฐฉ์งํ๊ณ , ์๋ก์ด ์ด๋ฏธ์ง๋ ์ ๋ถ๋ฅํ  ์ ์๊ฒ ๋ง๋ค์ด ์์ธก ๋ฒ์๋ ๋ํ์ค ์ ์์ต๋๋ค.  

์ด๋ฐ ์ ์ฒ๋ฆฌ ๊ณผ์ ์ ๋๊ธฐ ์ํด ์ผ๋ผ์ค๋ ImageDataGenerator ํด๋์ค๋ฅผ ์ ๊ณตํฉ๋๋ค. ImageDataGenerator๋ ์๋์ ๊ฐ์ ์ผ์ ํ  ์ ์์ต๋๋ค  

* ํ์ต ๊ณผ์ ์์ ์ด๋ฏธ์ง์ ์์ ๋ณํ ๋ฐ ์ ๊ทํ ์ ์ฉ  
* ๋ณํ๋ ์ด๋ฏธ์ง๋ฅผ ๋ฐฐ์น ๋จ์๋ก ๋ถ๋ฌ์ฌ ์ ์๋ generator ์์ฑ  
- generator๋ฅผ ์์ฑํ  ๋ flow(data, labels), flow_from_directory(directory) ๋ ๊ฐ์ง ํจ์๋ฅผ ์ฌ์ฉ ํ  ์ ์์ต๋๋ค.  
- fit_generator(fit), evaluate_generator ํจ์๋ฅผ ์ฌ์ฉํ์ฌ generator๋ก ์ด๋ฏธ์ง๋ฅผ ๋ถ๋ฌ์ ๋ชจ๋ธ์ ํ์ต์ํค๊ณ  ํ๊ฐ ํ  ์ ์์ต๋๋ค.  
  
์ด๋ฏธ์ง ๋ฐ์ดํฐ ์์ฑ  

ImageDataGenerator๋ฅผ ํตํด์ ๋ฐ์ดํฐ๋ฅผ ๋ง๋ค์ด์ค ๊ฒ์๋๋ค.   

์ด๋ค ๋ฐฉ์์ผ๋ก ๋ฐ์ดํฐ๋ฅผ ์ฆ์์ํฌ ๊ฒ์ธ์ง ์๋์ ๊ฐ์ ์ต์์ ํตํด์ ์ค์ ํฉ๋๋ค.    

์ฐธ๊ณ ๋ก, augmentation์ train ๋ฐ์ดํฐ์๋ง ์ ์ฉ์์ผ์ผ ํ๊ณ , validation ๋ฐ test ์ด๋ฏธ์ง๋ augmentation์ ์ ์ฉํ์ง ์์ต๋๋ค.  

๋ชจ๋ธ ์ฑ๋ฅ์ ํ๊ฐํ  ๋์๋ ์ด๋ฏธ์ง ์๋ณธ์ ์ฌ์ฉํด์ผ ํ๊ธฐ์ rescale๋ง ์ ์ฉํด ์ ๊ทํํ๊ณ  ์งํํฉ๋๋ค  

<code>
# ์ด๋ฏธ์ง ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
    #train์์๋ง ์ ์ฉ
    train_datagen = ImageDataGenerator(rescale = 1./255, # ๋ชจ๋  ์ด๋ฏธ์ง ์์๊ฐ๋ค์ 255๋ก ๋๋๊ธฐ  
                                    rotation_range=25, # 0~25๋ ์ฌ์ด์์ ์์์ ๊ฐ๋๋ก ์๋ณธ์ด๋ฏธ์ง๋ฅผ ํ์   
                                    width_shift_range=0.05, # 0.05๋ฒ์ ๋ด์์ ์์์ ๊ฐ๋งํผ ์์์ ๋ฐฉํฅ์ผ๋ก ์ข์ฐ ์ด๋  
                                    height_shift_range=0.05, # 0.05๋ฒ์ ๋ด์์ ์์์ ๊ฐ๋งํผ ์์์ ๋ฐฉํฅ์ผ๋ก ์ํ ์ด๋  
                                    zoom_range=0.2, # (1-0.2)~(1+0.2) => 0.8~1.2 ์ฌ์ด์์ ์์์ ์์น๋งํผ ํ๋/์ถ์  
                                    horizontal_flip=True, # ์ข์ฐ๋ก ๋ค์ง๊ธฐ                                     
                                    vertical_flip=True,  
                                    fill_mode='nearest'  
                                    )   
    #validation ๋ฐ test ์ด๋ฏธ์ง๋ augmentation์ ์ ์ฉํ์ง ์๋๋ค;  
    #๋ชจ๋ธ ์ฑ๋ฅ์ ํ๊ฐํ  ๋์๋ ์ด๋ฏธ์ง ์๋ณธ์ ์ฌ์ฉ (rescale๋ง ์งํ)  
    validation_datagen = ImageDataGenerator(rescale = 1./255)  
    test_datagen = ImageDataGenerator(rescale = 1./255)   

</code>  

์ด๋ฏธ์ง ๋ฐ์ดํฐ ์๊ฐ ์ ์ด์, batch_size๋ฅผ ๊ฒฐ์ ํ๋ ๊ฒ์ ์ฌ๋ฌ ์ํ์ฐฉ์ค์ ์ด๋ ค์์ด ์์๊ฒ์ด๋ผ๊ณ  ์๊ฐํ์ต๋๋ค.  

Generator ์์ฑ์ batch_size์ steps_per_epoch(model fitํ  ๋)๋ฅผ ๊ณฑํ ๊ฐ์ด ํ๋ จ ์ํ ์ ๋ณด๋ค ์๊ฑฐ๋ ๊ฐ์์ผ ํฉ๋๋ค.   

์ด์ ๋ง์ถฐ, flow_from_directory() ์ต์์์ batch_size์ model fit()/fit_generator() ์ต์์ steps_per_epoch ๊ฐ์ ์กฐ์ ํด ๊ฐ๋ฉฐ ํ์ต์ ์๋ํ์์ต๋๋ค.  

<code>
    #flow_from_directory() ๋ฉ์๋๋ฅผ ์ด์ฉํด์ ํ๋ จ๊ณผ ํ์คํธ์ ์ฌ์ฉ๋  ์ด๋ฏธ์ง ๋ฐ์ดํฐ๋ฅผ ๋ง๋ค๊ธฐ
    #๋ณํ๋ ์ด๋ฏธ์ง ๋ฐ์ดํฐ ์์ฑ
    train_generator = train_datagen.flow_from_directory(train_dir,   
                                                        batch_size=16, # ํ๋ฒ์   ๋ณํ๋ ์ด๋ฏธ์ง 16๊ฐ์ฉ   ๋ง๋ค์ด๋ผ ๋ผ๋ ๊ฒ  
                                                        color_mode='rgba', # ํ๋ฐฑ   ์ด๋ฏธ์ง ์ฒ๋ฆฌ  
                                                        class_mode='categorical',   
                                                        target_size=(150,150)) #   target_size์ ๋ง์ถฐ์   ์ด๋ฏธ์ง์ ํฌ๊ธฐ๊ฐ ์กฐ์ ๋๋ค  
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
    #์ฐธ๊ณ ๋ก, generator ์์ฑ์ batch_size x steps_per_epoch (model fit์์) <= ํ๋ จ ์ํ ์ ๋ณด๋ค ์๊ฑฐ๋ ๊ฐ์์ผ ํ๋ค.  

</code>  
  
<code>
    # class ํ์ธ  
    train_generator.class_indices  
</code>
  
๋ชจ๋ธ ๊ตฌ์ฑ  

ํฉ์ฑ๊ณฑ ์ ๊ฒฝ๋ง ๋ชจ๋ธ์ ๊ตฌ์ฑํฉ๋๋ค.  

![image](https://user-images.githubusercontent.com/77331459/194784951-18705042-9f54-43c5-9982-4844afe8e629.png)  
  
๋ชจ๋ธ ํ์ต  

๋ชจ๋ธ ์ปดํ์ผ ๋จ๊ณ์์๋ compile() ๋ฉ์๋๋ฅผ ์ด์ฉํด์ ์์ค ํจ์(loss function)์ ์ตํฐ๋ง์ด์ (optimizer)๋ฅผ ์ง์ ํฉ๋๋ค.  

* ์์ค ํจ์๋ก โbinary_crossentropyโ๋ฅผ ์ฌ์ฉํ์ต๋๋ค.(๋ณ๊ฒฝ ์์ )
* ๋ํ, ์ตํฐ๋ง์ด์ ๋ก๋ RMSprop์ ์ฌ์ฉํ์ต๋๋ค. RMSprop(Root Mean Square Propagation) ์๊ณ ๋ฆฌ์ฆ์ ํ๋ จ ๊ณผ์  ์ค์ ํ์ต๋ฅ ์ ์ ์ ํ๊ฒ ๋ณํ์์ผ ์ค๋๋ค.  
* ํ๋ จ๊ณผ ํ์คํธ๋ฅผ ์ํ ๋ฐ์ดํฐ์์ธ train_generator, validation_generator๋ฅผ ์๋ ฅํฉ๋๋ค.
* epochs๋ ๋ฐ์ดํฐ์์ ํ ๋ฒ ํ๋ จํ๋ ๊ณผ์ ์ ์๋ฏธํฉ๋๋ค.
* steps_per_epoch๋ ํ ๋ฒ์ ์ํฌํฌ (epoch)์์ ํ๋ จ์ ์ฌ์ฉํ  ๋ฐฐ์น (batch)์ ๊ฐ์๋ฅผ ์ง์ ํฉ๋๋ค.
* validation_steps๋ ํ ๋ฒ์ ์ํฌํฌ๊ฐ ๋๋  ๋, ํ์คํธ์ ์ฌ์ฉ๋๋ ๋ฐฐ์น (batch)์ ๊ฐ์๋ฅผ ์ง์ ํฉ๋๋ค.

<code>  
    from tensorflow.keras.optimizers import RMSprop  
  
    #compile() ๋ฉ์๋๋ฅผ ์ด์ฉํด์ ์์ค ํจ์ (loss function)์ ์ตํฐ๋ง์ด์  (optimizer)๋ฅผ ์ง์   
    model.compile(optimizer=RMSprop(learning_rate=0.001), # ์ตํฐ๋ง์ด์ ๋ก๋ RMSprop ์ฌ์ฉ  
                loss='binary_crossentropy', # ์์ค ํจ์๋ก โsparse_categorical_crossentropyโ ์ฌ์ฉ  
                metrics= ['accuracy'])  
    # RMSprop (Root Mean Square Propagation) Algorithm: ํ๋ จ ๊ณผ์  ์ค์ ํ์ต๋ฅ ์ ์ ์ ํ๊ฒ ๋ณํ์ํจ๋ค.  
  
  
  
    #๋ชจ๋ธ ํ๋ จ  
    history = model.fit_generator(train_generator, # train_generator์์ X๊ฐ, y๊ฐ ๋ค ์์ผ๋ generator๋ง ์ฃผ๋ฉด ๋๋ค  
                                validation_data=validation_generator, # validatino_generator์์๋ ๊ฒ์ฆ์ฉ X,y๋ฐ์ดํฐ๋ค์ด ๋ค ์์ผ๋ generator๋ก ์ฃผ๋ฉด ๋จ  
                                steps_per_epoch=4, # ํ ๋ฒ์ ์ํฌํฌ(epoch)์์ ํ๋ จ์ ์ฌ์ฉํ  ๋ฐฐ์น(batch)์ ๊ฐ์ ์ง์ ; generator๋ฅผ 4๋ฒ ๋ถ๋ฅด๊ฒ ๋ค  
                                epochs=100, # ๋ฐ์ดํฐ์์ ํ ๋ฒ ํ๋ จํ๋ ๊ณผ์ ; epoch์ 100 ์ด์์ ์ค์ผํ๋ค  
                                validation_steps=4, # ํ ๋ฒ์ ์ํฌํฌ๊ฐ ๋๋  ๋, ๊ฒ์ฆ์ ์ฌ์ฉ๋๋ ๋ฐฐ์น(batch)์ ๊ฐ์๋ฅผ ์ง์ ; validation_generator๋ฅผ 4๋ฒ ๋ถ๋ฌ์ ๋์จ ์ด๋ฏธ์ง๋ค๋ก ์์์ ํด๋ผ  
                                verbose=2)  
    #์ฐธ๊ณ : validation_steps๋ ๋ณดํต ๋ด๊ฐ ์ํ๋ ์ด๋ฏธ์ง ์์ flowํ  ๋ ์ง์ ํ batchsize๋ก ๋๋ ๊ฐ์ validation_steps๋ก ์ง์   

</code>
  
  
๊ฒฐ๊ณผ ํ์ธ ๋ฐ ํ๊ฐ  
ํ์ต๋ ๋ชจ๋ธ ๊ฒฐ๊ณผ์ ์ฑ๋ฅ์ ํ์ธํฉ๋๋ค.  

<code>

    # ๋ชจ๋ธ ์ฑ๋ฅ ํ๊ฐ  
    model.evaluate(train_generator)  
</code>  

<code>

    model.evaluate(validation_generator)        
</code>  
  
    
์ ํ๋ ๋ฐ ์์ค ์๊ฐํ  

ํ๋ จ ๊ณผ์ ์์ epoch์ ๋ฐ๋ฅธ ์ ํ๋์ ์์ค์ ์๊ฐํํ์ฌ ํ์ธํฉ๋๋ค.  

<code>

    # ์ ํ๋ ๋ฐ ์์ค ์๊ฐํ  
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
  
ํ์คํธ ํ๊ฐ  
#

- datagenImage1 

- epoch 50 , patience=5
batchsize 4
iterations 5

ํ์คํธ์ฉ
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

ํ๋ จ์ฉ(์ค์ ์ ์์ ๋์ผ)</br>
trainImage folder = vsCode\\PillProject\\imageT\\color\\train</br>

batch_size = 64</br>
num_classes = 5</br>
epochs = 50</br>


<img width="373" alt="datagenImage1ํ์ต์ด๋ฏธ์ง(patience=5)" src="https://user-images.githubusercontent.com/77331459/205564571-cd6c7f15-97ae-4b2a-bdf4-6480d578ab82.png">

์ ๊ฒฝ๋ง ๊ตฌ์ฑ

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

18์์ ํ์ต์ค์ง (์ต๋ 50)</br>
Loss : 0.7770828008651733, Acc : 0.8653846383094788</br>

<img width="251" alt="datagenImage1๊ฒฐ๊ณผ(patience=5)" src="https://user-images.githubusercontent.com/77331459/205564584-c17a2b86-9722-4c62-953a-81a6a01105e8.png">

<img width="454" alt="datagenImage1๊ฒ์ฆ(patience=5)" src="https://user-images.githubusercontent.com/77331459/205564559-34f99053-f66f-443a-a9f5-1c9966a490b2.png">

#

- datagenImage1 

- epoch 50 , patience = 10
batchsize 4
iterations 5

ํ์คํธ์ฉ
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

ํ๋ จ์ฉ(์ค์ ์ ์์ ๋์ผ)</br>
trainImage folder = vsCode\\PillProject\\imageT\\color\\train</br>

batch_size = 64</br>
num_classes = 5</br>
epochs = 50</br>

<img width="370" alt="datagenImage1ํ์ต์ด๋ฏธ์ง(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564671-0d98e75f-9e3c-4e3b-b054-6f1b21ad5345.png">

์ ๊ฒฝ๋ง ๊ตฌ์ฑ

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

30์์ ํ์ต์ค์ง (์ต๋ 50)</br>
Loss : 2.2101550102233887, Acc : 0.699999988079071</br>

<img width="248" alt="datagenImage1๊ฒฐ๊ณผ(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564680-ce4f6eaa-4a2a-4025-b0af-ccfc39b8e297.png">

<img width="465" alt="datagenImage1๊ฒ์ฆ(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564686-247119cc-75ac-4ded-a589-ed70fddf95f7.png">

#

- datagenImage1 

- epoch 50 , patience = 20</br>
batchsize 4 </br>
iterations 5 </br>

ํ์คํธ์ฉ
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

ํ๋ จ์ฉ(์ค์ ์ ์์ ๋์ผ)</br>
trainImage folder = vsCode\\PillProject\\imageT\\color\\train</br>

batch_size = 64</br>
num_classes = 5</br>
epochs = 50</br>

<img width="374" alt="datagenImage1ํ์ต์ด๋ฏธ์ง(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564637-c238ab2e-3059-48a1-8157-f7089819c56d.png">

์ ๊ฒฝ๋ง ๊ตฌ์ฑ

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

50๊น์ง ํ์ต ์๋ฃ (์ต๋ 50)</br>
Loss : 0.7318955659866333, Acc : 1.0</br>

<img width="259" alt="datagenImage1๊ฒฐ๊ณผ(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564626-ada0b971-d632-4b77-acfe-53f874463ee4.png">

<img width="460" alt="datagenImage1๊ฒ์ฆ(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564646-fbd46928-91fa-4499-a46d-ccf3cc10869c.png">

#

- datagenImage2

- epoch 50 , patience = 10</br>
batchsize 4 </br>

ํ๋ จ์ฉ
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

ํ์คํธ์ฉ(๊ฒฝ๋ก ์ ์ธ ์์ ๋์ผ)</br>
C:\\vsCode\\PillProject\\imageT\\color\\train</br>

num_classes = 5</br>
epochs = 50</br>

<img width="369" alt="datagenImage2ํ์ต์ด๋ฏธ์ง(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564721-7181cb4f-7bc7-4973-b93c-859fb2358b7c.png">

์ ๊ฒฝ๋ง ๊ตฌ์ฑ

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

35๊น์ง ํ์ต ์๋ฃ (์ต๋ 50)</br>
Loss : 0.023338522762060165, Acc : 1.0</br>

<img width="261" alt="datagenImage2๊ฒฐ๊ณผ(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564730-979169c2-b521-4756-8a78-cd2607ae81f6.png">

<img width="453" alt="datagenImage2๊ฒ์ฆ(patience=10)" src="https://user-images.githubusercontent.com/77331459/205564736-e0b6c974-7fa4-43c3-8844-be7b84df4239.png">

#

- datagenImage2

- epoch 50 , patience = 20</br>
batchsize 4 </br>

ํ๋ จ์ฉ
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

ํ์คํธ์ฉ(๊ฒฝ๋ก ์ ์ธ ์์ ๋์ผ)</br>
C:\\vsCode\\PillProject\\imageT\\color\\train</br>

num_classes = 5</br>
epochs = 50</br>

<img width="367" alt="datagenImage2ํ์ต์ด๋ฏธ์ง(patience = 20)" src="https://user-images.githubusercontent.com/77331459/205564745-cd704b5f-ca7a-49de-a24c-020d6dc42212.png">

์ ๊ฒฝ๋ง ๊ตฌ์ฑ

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

35๊น์ง ํ์ต ์๋ฃ (์ต๋ 50)</br>
Loss : 0.38273733854293823, Acc : 0.921875</br>

<img width="270" alt="datagenImage2๊ฒฐ๊ณผ(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564753-0a10c105-c897-466a-9205-afb2cf4d8f00.png">

<img width="441" alt="datagenImage2๊ฒ์ฆ(patience=20)" src="https://user-images.githubusercontent.com/77331459/205564767-ff55c2d3-d4a8-40ea-bdc1-892d98150114.png">


#

- datagenImage2

- epoch 50 , patience = 30</br>
batchsize 4 </br>

ํ๋ จ์ฉ
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

ํ์คํธ์ฉ(๊ฒฝ๋ก ์ ์ธ ์์ ๋์ผ)</br>
C:\\vsCode\\PillProject\\imageT\\color\\train</br>

num_classes = 5</br>
epochs = 50</br>

<img width="370" alt="datagenImage2ํ์ต์ด๋ฏธ์ง(patience = 30)" src="https://user-images.githubusercontent.com/77331459/205564801-139e2d6a-954c-4bb2-89bd-6bc7a8867203.png">

์ ๊ฒฝ๋ง ๊ตฌ์ฑ

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

50๊น์ง ํ์ต ์๋ฃ (์ต๋ 50)</br>
Loss : 0.10279396176338196, Acc : 0.953125</br>

<img width="267" alt="datagenImage2๊ฒฐ๊ณผ(patience=30)" src="https://user-images.githubusercontent.com/77331459/205564843-c8d2d900-d0bf-4a5a-a9f3-a753319f9584.png">

<img width="468" alt="datagenImage2๊ฒ์ฆ(patience=30)" src="https://user-images.githubusercontent.com/77331459/205564865-b575dfc7-3d21-4941-8d24-88d37f21bdca.png">


# Shape  
color์ ๋์ผํ cnn๋ชจ๋ธ์ ์ฌ์ฉํ์์

ํ์คํธ ํ๊ฐ
#

epoch 5000  |  patience 100</br>

batch_size = 4</br>
#์์ฑ์์ ํ๋ผ๋ฏธํฐ๋ฅผ ์ค์ ํ๋ฉด ์ด๋ป๊ฒ augmentation๋ฅผ ์งํํ ์ง ์ง์ ํ  ์ ์๋ค.</br>
<code>

    datagen = ImageDataGenerator(
        featurewise_center = True)
</code>

ํ์คํธ์ฉ</br>

<code>

    #๊ฒฝ๋ก, ์ํ, ์ด๋ฏธ์ง์ฌ์ด์ฆ, ํ๋ฒ์ ์ฝ์ด์ฌ ์ด๋ฏธ์ง ์, ํด๋์ค ๋ชจ๋
    generator = datagen.flow_from_directory(
        'C:\\vsCode\\PillProject\\imageT\\shape\\test', 
        shuffle = True, 
        target_size=(256,256), 
        batch_size=batch_size, 
        class_mode = 'categorical',
        color_mode='grayscale')
    ํ๋ จ์ฉ(๊ฒฝ๋ก๋ง ๋ค๋ฆ)
    C:\\vsCode\\PillProject\\imageT\\shape\\train

    class_names = ['circle', 'hexagon', 'pentagon', 'rectangle', 'rectangular', 'triangle']
</code>

<img width="366" alt="shapeDatagenImage1ํ์ต์ด๋ฏธ์ง(patience = 100)" src="https://user-images.githubusercontent.com/77331459/205568304-b21fb1b0-7542-44fe-ad87-db7da1fb1fd9.png">

#๋ฐฐ์น ์ฌ์ด์ฆ์ ์๋งํผ ์ด๋ฏธ์ง๋ฅผ ํ์ตํ๊ณ  ๊ฐ์ค์น๋ฅผ ๊ฐฑ์ ํ๊ฒ๋๋ค.</br>
#๋ฐฐ์น ์ฌ์ด์ฆ๋ฅผ ์ฆ๊ฐ์ํค๋ฉด ํ์ํ ๋ฉ๋ชจ๋ฆฌ๊ฐ ์ฆ๊ฐํ๋ ๋ชจ๋ธ์ ํ๋ จํ๋๋ฐ ์๊ฐ์ด ์ ๊ฒ ๋ ๋ค.</br>
#๋ฐฐ์น ์ฌ์ด์ฆ๋ฅผ ๊ฐ์์ํค๋ฉด ํ์ํ ๋ฉ๋ชจ๋ฆฌ๊ฐ ๊ฐ์ํ๋ ๋ชจ๋ธ์ ํ๋ จํ๋๋ฐ ์๊ฐ์ด ๋ง์ด ๋ ๋ค.</br>
batch_size = 64</br>
#๋ถ๋ฅ๋  ํด๋์ค ๊ฐ์</br>
num_classes = 6</br>
#๋ช๋ฒ ํ์ต์ ๋ฐ๋ณตํ  ๊ฒ์ธ์ง ๊ฒฐ์ </br>
#์ํฌํฌ๊ฐ ๋ง๋ค๋ฉด ๊ณผ์ ํฉ ๋ฌธ์  ๋ฐ์๊ฐ๋ฅ, ์ ๋ค๋ฉด ๋ถ๋ฅ๋ฅผ ์ ๋๋ก ๋ชปํ  ์ ์๋ค.</br>
epochs = 5000</br>

#๋ชจ๋ธ ๊ตฌ์ฑ
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

#๋ชจ๋ธ์ ํ์ต์ํค๊ธฐ ์  ํ๊ฒฝ ์ค์  (์ ๊ทํ๊ธฐ, ์์คํจ์, ํ๊ฐ์งํ)</br>

#์ ๊ทํ๊ธฐ - ํ๋ จ๊ณผ์ ์ ์ค์ ํฉ๋๋ค. ์ฆ, ์ต์ ํ ์๊ณ ๋ฆฌ์ฆ์ ์ค์ ์ ์๋ฏธํฉ๋๋ค.</br>
#adam, sgd, rmsprop, adagrad ๋ฑ์ด ์์ต๋๋ค.</br>
#๋ถ๋ฅ์๋  โSGDโ, โAdamโ, โRMSpropโ</br>

#์์คํจ์ - ๋ชจ๋ธ์ด ์ต์ ํ์ ์ฌ์ฉ๋๋ ๋ชฉ์  ํจ์์๋๋ค.</br>
#mse, categorical_crossentropy, binary_crossentropy ๋ฑ์ด ์์ต๋๋ค.</br>

#ํ๊ฐ์งํ - ํ๋ จ์ ๋ชจ๋ํฐ๋ง ํ๊ธฐ ์ํด ์ฌ์ฉ๋ฉ๋๋ค.</br>
#๋ถ๋ฅ์์๋ accuracy, ํ๊ท์์๋ mse, rmse, r2, mae, mspe, mape, msle ๋ฑ์ด ์์ต๋๋ค.</br>
#์ฌ์ฉ์๊ฐ ๋ฉํธ๋ฆญ์ ์ ์ํด์ ์ฌ์ฉํ  ์๋ ์์ต๋๋ค.</br>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )
</code>

#๊ณผ์ ํฉ์ ๋ฐฉ์งํ๊ธฐ ์ํด์ ์ค์ </br>
#Early Stopping ์ด๋ ๋๋ฌด ๋ง์ Epoch ์ overfitting ์ ์ผ์ผํจ๋ค.ย ํ์ง๋ง ๋๋ฌด ์ ์ Epoch ์ underfitting ์ ์ผ์ผํจ๋ค.ย </br>
#๋ผ๋ ๋๋ ๋ง๋ฅผ ํด๊ฒฐํ๊ธฐ์ํจ</br>

#Epoch ์ ์ ํ๋๋ฐ ๋ง์ด ์ฌ์ฉ๋๋ Early stopping ์ ๋ฌด์กฐ๊ฑด Epoch ์ ๋ง์ด ๋๋ฆฐ ํ, ํน์  ์์ ์์ ๋ฉ์ถ๋ ๊ฒ์ด๋ค.ย </br>
#๊ทธ ํน์ ์์ ์ ์ด๋ป๊ฒ ์ ํ๋๋๊ฐ Early stopping ์ ํต์ฌ์ด๋ผ๊ณ  ํ  ์ ์๋ค. </br>
#์ผ๋ฐ์ ์ผ๋ก hold-out validation set ์์์ ์ฑ๋ฅ์ด ๋์ด์ ์ฆ๊ฐํ์ง ์์ ๋ย ํ์ต์ ์ค์ง</br>
#https://3months.tistory.com/424 ์ฐธ๊ณ </br>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=100)
    #ModelCheckpoint instance๋ฅผ callbacks ํ๋ผ๋ฏธํฐ์ ๋ฃ์ด์ค์ผ๋ก์จ, ๊ฐ์ฅ validation performance ๊ฐ ์ข์๋ ๋ชจ๋ธ์ ์ ์ฅํ  ์ ์๊ฒ๋๋ค.
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        # ์ฝ๋ฐฑ์ ์ ์ฉํ๋ฉด ๊ณผ์ ํฉ์ ๋ฐฉ์งํ๋์ง๋ ์ ๋ชจ๋ฅด๊ฒ ์.
        # ๋ค๋ง, ์ ์ฉ์ํค๋ฉด ํ๋ จ์ 12~20๋ฒ ์ฌ์ด๋ก ํ๊ณ  ๊ทธ๋ฅ ์ํฌํฌ ์๊ด์์ด ํ๋ จ์ ์ค์งํจ.
        callbacks=[early_stopping,mc]
    )
</code>

119 ๊น์ง๋ง ํ์ต (์ต๋ 5000)</br>
Loss : 4.203685283660889, Acc : 0.5714285969734192</br>

<img width="267" alt="shapeDatagenImage1๊ฒฐ๊ณผ(patience = 100)" src="https://user-images.githubusercontent.com/77331459/205568319-e5876d66-b066-4396-a993-10d3c7638135.png">

<img width="476" alt="shapeDatagenImage1๊ฒ์ฆ(patience = 100)" src="https://user-images.githubusercontent.com/77331459/205568331-fe73dc87-0817-44b4-bd22-46755b7d734c.png">

#

epoch 5000  |  patience 50</br>

batch_size = 4</br>
#์์ฑ์์ ํ๋ผ๋ฏธํฐ๋ฅผ ์ค์ ํ๋ฉด ์ด๋ป๊ฒ augmentation๋ฅผ ์งํํ ์ง ์ง์ ํ  ์ ์๋ค.</br>
<code>

    datagen = ImageDataGenerator(
        featurewise_center = True)
</code>

ํ์คํธ์ฉ</br>

<code>

    #๊ฒฝ๋ก, ์ํ, ์ด๋ฏธ์ง์ฌ์ด์ฆ, ํ๋ฒ์ ์ฝ์ด์ฌ ์ด๋ฏธ์ง ์, ํด๋์ค ๋ชจ๋
    generator = datagen.flow_from_directory(
        'C:\\vsCode\\PillProject\\imageT\\shape\\test', 
        shuffle = True, 
        target_size=(256,256), 
        batch_size=batch_size, 
        class_mode = 'categorical',
        color_mode='grayscale')
    ํ๋ จ์ฉ(๊ฒฝ๋ก๋ง ๋ค๋ฆ)
    C:\\vsCode\\PillProject\\imageT\\shape\\train

    class_names = ['circle', 'hexagon', 'pentagon', 'rectangle', 'rectangular', 'triangle']
</code>

<img width="379" alt="shapeDatagenImage1ํ์ต์ด๋ฏธ์ง(patience = 50)" src="https://user-images.githubusercontent.com/77331459/205567847-482f8737-d5fa-4b1c-ab45-1ce391dc1b9e.png">

#๋ฐฐ์น ์ฌ์ด์ฆ์ ์๋งํผ ์ด๋ฏธ์ง๋ฅผ ํ์ตํ๊ณ  ๊ฐ์ค์น๋ฅผ ๊ฐฑ์ ํ๊ฒ๋๋ค.</br>
#๋ฐฐ์น ์ฌ์ด์ฆ๋ฅผ ์ฆ๊ฐ์ํค๋ฉด ํ์ํ ๋ฉ๋ชจ๋ฆฌ๊ฐ ์ฆ๊ฐํ๋ ๋ชจ๋ธ์ ํ๋ จํ๋๋ฐ ์๊ฐ์ด ์ ๊ฒ ๋ ๋ค.</br>
#๋ฐฐ์น ์ฌ์ด์ฆ๋ฅผ ๊ฐ์์ํค๋ฉด ํ์ํ ๋ฉ๋ชจ๋ฆฌ๊ฐ ๊ฐ์ํ๋ ๋ชจ๋ธ์ ํ๋ จํ๋๋ฐ ์๊ฐ์ด ๋ง์ด ๋ ๋ค.</br>
batch_size = 64</br>
#๋ถ๋ฅ๋  ํด๋์ค ๊ฐ์</br>
num_classes = 6</br>
#๋ช๋ฒ ํ์ต์ ๋ฐ๋ณตํ  ๊ฒ์ธ์ง ๊ฒฐ์ </br>
#์ํฌํฌ๊ฐ ๋ง๋ค๋ฉด ๊ณผ์ ํฉ ๋ฌธ์  ๋ฐ์๊ฐ๋ฅ, ์ ๋ค๋ฉด ๋ถ๋ฅ๋ฅผ ์ ๋๋ก ๋ชปํ  ์ ์๋ค.</br>
epochs = 5000</br>

#๋ชจ๋ธ ๊ตฌ์ฑ
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

#๋ชจ๋ธ์ ํ์ต์ํค๊ธฐ ์  ํ๊ฒฝ ์ค์  (์ ๊ทํ๊ธฐ, ์์คํจ์, ํ๊ฐ์งํ)</br>

#์ ๊ทํ๊ธฐ - ํ๋ จ๊ณผ์ ์ ์ค์ ํฉ๋๋ค. ์ฆ, ์ต์ ํ ์๊ณ ๋ฆฌ์ฆ์ ์ค์ ์ ์๋ฏธํฉ๋๋ค.</br>
#adam, sgd, rmsprop, adagrad ๋ฑ์ด ์์ต๋๋ค.</br>
#๋ถ๋ฅ์๋  โSGDโ, โAdamโ, โRMSpropโ</br>

#์์คํจ์ - ๋ชจ๋ธ์ด ์ต์ ํ์ ์ฌ์ฉ๋๋ ๋ชฉ์  ํจ์์๋๋ค.</br>
#mse, categorical_crossentropy, binary_crossentropy ๋ฑ์ด ์์ต๋๋ค.</br>

#ํ๊ฐ์งํ - ํ๋ จ์ ๋ชจ๋ํฐ๋ง ํ๊ธฐ ์ํด ์ฌ์ฉ๋ฉ๋๋ค.</br>
#๋ถ๋ฅ์์๋ accuracy, ํ๊ท์์๋ mse, rmse, r2, mae, mspe, mape, msle ๋ฑ์ด ์์ต๋๋ค.</br>
#์ฌ์ฉ์๊ฐ ๋ฉํธ๋ฆญ์ ์ ์ํด์ ์ฌ์ฉํ  ์๋ ์์ต๋๋ค.</br>

<code>

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy']
    )
</code>

#๊ณผ์ ํฉ์ ๋ฐฉ์งํ๊ธฐ ์ํด์ ์ค์ </br>
#Early Stopping ์ด๋ ๋๋ฌด ๋ง์ Epoch ์ overfitting ์ ์ผ์ผํจ๋ค.ย ํ์ง๋ง ๋๋ฌด ์ ์ Epoch ์ underfitting ์ ์ผ์ผํจ๋ค.ย </br>
#๋ผ๋ ๋๋ ๋ง๋ฅผ ํด๊ฒฐํ๊ธฐ์ํจ</br>

#Epoch ์ ์ ํ๋๋ฐ ๋ง์ด ์ฌ์ฉ๋๋ Early stopping ์ ๋ฌด์กฐ๊ฑด Epoch ์ ๋ง์ด ๋๋ฆฐ ํ, ํน์  ์์ ์์ ๋ฉ์ถ๋ ๊ฒ์ด๋ค.ย </br>
#๊ทธ ํน์ ์์ ์ ์ด๋ป๊ฒ ์ ํ๋๋๊ฐ Early stopping ์ ํต์ฌ์ด๋ผ๊ณ  ํ  ์ ์๋ค. </br>
#์ผ๋ฐ์ ์ผ๋ก hold-out validation set ์์์ ์ฑ๋ฅ์ด ๋์ด์ ์ฆ๊ฐํ์ง ์์ ๋ย ํ์ต์ ์ค์ง</br>
#https://3months.tistory.com/424 ์ฐธ๊ณ </br>

<code>

    early_stopping=EarlyStopping(monitor='val_loss', patience=50)
    #ModelCheckpoint instance๋ฅผ callbacks ํ๋ผ๋ฏธํฐ์ ๋ฃ์ด์ค์ผ๋ก์จ, ๊ฐ์ฅ validation performance ๊ฐ ์ข์๋ ๋ชจ๋ธ์ ์ ์ฅํ  ์ ์๊ฒ๋๋ค.
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        shuffle=True,
        # ์ฝ๋ฐฑ์ ์ ์ฉํ๋ฉด ๊ณผ์ ํฉ์ ๋ฐฉ์งํ๋์ง๋ ์ ๋ชจ๋ฅด๊ฒ ์.
        # ๋ค๋ง, ์ ์ฉ์ํค๋ฉด ํ๋ จ์ 12~20๋ฒ ์ฌ์ด๋ก ํ๊ณ  ๊ทธ๋ฅ ์ํฌํฌ ์๊ด์์ด ํ๋ จ์ ์ค์งํจ.
        callbacks=[early_stopping,mc]
    )
</code>

119 ๊น์ง๋ง ํ์ต (์ต๋ 5000)</br>
Loss : 4.203685283660889, Acc : 0.5714285969734192</br>

<img width="275" alt="shapeDatagenImage1๊ฒฐ๊ณผ(patience = 50)" src="https://user-images.githubusercontent.com/77331459/205567857-b39c44eb-e9e5-4aeb-8722-e19768207f47.png">

<img width="461" alt="shapeDatagenImage1๊ฒ์ฆ(patience = 50)" src="https://user-images.githubusercontent.com/77331459/205567868-e0f25d31-cce9-4db9-8383-b6472402189b.png">

# String  
![๋ค์ด๋ก๋ (3)](https://user-images.githubusercontent.com/77331459/194784410-d8690c98-46e6-429f-8125-36897550d5d6.png)  

OCR์ Optical Character Recognition์ ์ฝ์๋ก ์ฌ๋์ด ์ฐ๊ฑฐ๋ ๊ธฐ๊ณ๋ก ์ธ์ํ ๋ฌธ์์ ์์์ ์ด๋ฏธ์ง ์ค์บ๋๋ก ํ๋ํ์ฌ ๊ธฐ๊ณ๊ฐ ์ฝ์ ์ ์๋ ๋ฌธ์๋ก ๋ณํํ๋ ๊ฒ์ ๋ปํ๋ค.  

ํ์ด์ฌ์์ ์ฌ์ฉํ  ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ pytesseract์ด๋ค.  

<code>
ํ์๋ํธ(Tesseract)๋ ๋ค์ํ ์ด์ ์ฒด์ ๋ฅผ ์ํ ๊ดํ ๋ฌธ์ ์ธ์ ์์ง์ด๋ค. ์ด ์ํํธ์จ์ด๋ Apache License, ๋ฒ์  2.0,์ ๋ฐ๋ผ ๋ฐฐํฌ๋๋ ๋ฌด๋ฃ ์ํํธ์จ์ด์ด๋ฉฐ 2006๋๋ถํฐ Google์์ ๊ฐ๋ฐ์ ํ์ํ๋ค.
</code>  

๊ธฐ๋ฅ

* get_languages Tesseract OCR์์ ํ์ฌ ์ง์ํ๋ ๋ชจ๋  ์ธ์ด๋ฅผ ๋ฐํํฉ๋๋ค. 
* get_tesseract_version ์์คํ์ ์ค์น๋ Tesseract ๋ฒ์ ์ ๋ฐํํฉ๋๋ค.
* image_to_string Tesseract OCR ์ฒ๋ฆฌ์์ ์์ ๋์ง ์์ ์ถ๋ ฅ์ ๋ฌธ์์ด๋ก ๋ฐํํฉ๋๋ค.
* image_to_boxes ์ธ์ ๋ ๋ฌธ์์ ํด๋น ์์ ๊ฒฝ๊ณ๋ฅผ ํฌํจํ๋ ๊ฒฐ๊ณผ๋ฅผ ๋ฐํํฉ๋๋ค.
* image_to_data ์์ ๊ฒฝ๊ณ, ์ ๋ขฐ๋ ๋ฐ ๊ธฐํ ์ ๋ณด๊ฐ ํฌํจ ๋ ๊ฒฐ๊ณผ๋ฅผ ๋ฐํํฉ๋๋ค. Tesseract 3.05 ์ด์์ด ํ์ํฉ๋๋ค. ์์ธํ ๋ด์ฉ์ Tesseract TSV ๋ฌธ์ ๋ฅผ ํ์ธํ์ญ์์ค.
* image_to_osd ๋ฐฉํฅ ๋ฐ ์คํฌ๋ฆฝํธ ๊ฐ์ง์ ๋ํ ์ ๋ณด๊ฐ ํฌํจ ๋ ๊ฒฐ๊ณผ๋ฅผ ๋ฐํํฉ๋๋ค.
* image_to_alto_xml Tesseract์ ALTO XML ํ์์ ํ์์ผ๋ก ๊ฒฐ๊ณผ๋ฅผ ๋ฐํํฉ๋๋ค.
* run_and_get_output Tesseract OCR์์ ์์ ์ถ๋ ฅ์ ๋ฐํํฉ๋๋ค. tesseract๋ก ์ ์ก๋๋ ๋งค๊ฐ ๋ณ์๋ฅผ ์ข ๋ ์ ์ด ํ  ์ ์์ต๋๋ค.

๋งค๊ฐ ๋ณ์

* image ๊ฐ์ฒด ๋๋ ๋ฌธ์์ด-Tesseract์์ ์ฒ๋ฆฌ ํ  ์ด๋ฏธ์ง์ PIL ์ด๋ฏธ์ง / NumPy ๋ฐฐ์ด ๋๋ ํ์ผ ๊ฒฝ๋ก์๋๋ค. ํ์ผ ๊ฒฝ๋ก ๋์  ๊ฐ์ฒด๋ฅผ ์ ๋ฌํ๋ฉด pytesseract๋ ์์ ์ ์ผ๋ก ์ด๋ฏธ์ง๋ฅผ RGB ๋ชจ๋ ๋ก ๋ณํ ํฉ๋๋ค .
* lang String-Tesseract ์ธ์ด ์ฝ๋ ๋ฌธ์์ด์๋๋ค. ์ง์ ๋์ง ์์ ๊ฒฝ์ฐ ๊ธฐ๋ณธ๊ฐ์ eng์๋๋ค ! ์ฌ๋ฌ ์ธ์ด์ ์ : lang = 'eng + fra'
* config String- pytesseract ํจ์๋ฅผ ํตํด ์ฌ์ฉํ  ์์๋ ์ถ๊ฐ ์ฌ์ฉ์ ์ง์  ๊ตฌ์ฑ ํ๋๊ทธ ์๋๋ค. ์ : config = '-psm 6'
* nice Integer-Tesseract ์คํ์ ๋ํ ํ๋ก์ธ์ ์ฐ์  ์์๋ฅผ ์์ ํฉ๋๋ค. Windows์์๋ ์ง์๋์ง ์์ต๋๋ค. Nice๋ ์ ๋์ค์ ์ ์ฌํ ํ๋ก์ธ์ค์ ์ฐ์์ฑ์ ์กฐ์ ํฉ๋๋ค.
* output_type ํด๋์ค ์์ฑ-์ถ๋ ฅ ์ ํ์ ์ง์ ํ๋ฉฐ ๊ธฐ๋ณธ๊ฐ์ string ์๋๋ค. ์ง์๋๋ ๋ชจ๋  ์ ํ์ ์ ์ฒด ๋ชฉ๋ก์ pytesseract.Output ํด๋์ค ์ ์ ์๋ฅผ ํ์ธํ์ธ์ .
* timeout Integer ๋๋ Float-OCR ์ฒ๋ฆฌ๋ฅผ์ํ ๊ธฐ๊ฐ (์ด). ๊ทธ ํ pytesseract๊ฐ ์ข๋ฃ๋๊ณ  RuntimeError๊ฐ ๋ฐ์ํฉ๋๋ค.
* pandas_config Dict- Output.DATAFRAME ์ ํ ์๋ง ํด๋น๋ฉ๋๋ค . pandas.read_csv์ ๋ํ ์ฌ์ฉ์ ์ง์  ์ธ์๊ฐ์๋ ์ฌ์  . image_to_data ์ ์ถ๋ ฅ์ ์ฌ์ฉ์ ์ ์ ํ  ์ ์์ต๋๋ค .



# ์ฌ์ฉํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ
* yolov5

color์์ ์ฌ์ฉํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ
* numpy
* cv2
* KMeans
* matplotlib
* PIL
* os

colorCss์์ ์ฌ์ฉํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ
* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* os
* PIL
* shutil

shape์์ ์ฌ์ฉํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ
* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* os
* PIL
* shutil

string์์ ์ฌ์ฉํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ
* Image
* pytesseract

# Commit ๊ท์น
> ์ปค๋ฐ ์ ๋ชฉ์ ์ต๋ 50์ ์๋ ฅ  

๋ณธ๋ฌธ์ ํ ์ค ์ต๋ 72์ ์๋ ฅ  

Commit ๋ฉ์ธ์ง  


๐ช[chore]: ์ฝ๋ ์์ , ๋ด๋ถ ํ์ผ ์์ .  

โจ[feat]: ์๋ก์ด ๊ธฐ๋ฅ ๊ตฌํ.  

๐จ[style]: ์คํ์ผ ๊ด๋ จ ๊ธฐ๋ฅ.(์ฝ๋์ ๊ตฌ์กฐ/ํํ ๊ฐ์ )  

โ[add]: Feat ์ด์ธ์ ๋ถ์์ ์ธ ์ฝ๋ ์ถ๊ฐ, ๋ผ์ด๋ธ๋ฌ๋ฆฌ ์ถ๊ฐ  

๐ง[file]: ์๋ก์ด ํ์ผ ์์ฑ, ์ญ์  ์  

๐[fix]: ๋ฒ๊ทธ, ์ค๋ฅ ํด๊ฒฐ.  

๐ฅ[del]: ์ธ๋ชจ์๋ ์ฝ๋/ํ์ผ ์ญ์ .  

๐[docs]: README๋ WIKI ๋ฑ์ ๋ฌธ์ ๊ฐ์ .  

๐[mod]: storyboard ํ์ผ,UI ์์ ํ ๊ฒฝ์ฐ.  

โ๏ธ[correct]: ์ฃผ๋ก ๋ฌธ๋ฒ์ ์ค๋ฅ๋ ํ์์ ๋ณ๊ฒฝ, ์ด๋ฆ ๋ณ๊ฒฝ ๋ฑ์ ์ฌ์ฉํฉ๋๋ค.  

๐[move]: ํ๋ก์ ํธ ๋ด ํ์ผ์ด๋ ์ฝ๋(๋ฆฌ์์ค)์ ์ด๋  

โช๏ธ[rename]: ํ์ผ ์ด๋ฆ ๋ณ๊ฒฝ์ด ์์ ๋ ์ฌ์ฉํฉ๋๋ค.  

โก๏ธ[improve]: ํฅ์์ด ์์ ๋ ์ฌ์ฉํฉ๋๋ค.  

โป๏ธ[refactor]: ์ ๋ฉด ์์ ์ด ์์ ๋ ์ฌ์ฉํฉ๋๋ค.  

๐[merge]: ๋ค๋ฅธ๋ธ๋ ์น๋ฅผ merge ํ  ๋ ์ฌ์ฉํฉ๋๋ค.  

โ [test]: ํ์คํธ ์ฝ๋๋ฅผ ์์ฑํ  ๋ ์ฌ์ฉํฉ๋๋ค.  
