import cv2
from tensorflow import keras
import cv2 as cv #معالجة الصور المستخدمة
import numpy as np # العمليات الحسابية التي تقوم بها الالة
import matplotlib.pyplot as plt # عرض الصور والرسومات والنتائج
from tensorflow.keras import datasets, layers, models # لبناء و لتدريب النموذج الالي
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.python.layers.convolutional import Conv2D

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
#هنا قمنا بتحميل صور من 10 اشياء كما مكتوبة كل صورة تتكون من 32 بكسل وقام الموقع سي فار10 بتدريب النموذج على 50 الف صورة
#--واختباره على 10 الاف صورة
training_images, testing_images = training_images / 255.0, testing_images / 255.0
#هنا قمنا بتضييق النطاق من 255-0 الى 1-0 لان الالة تتعامل بشكل افضل مع الارقام الصغيرة
class_names = ['Plane', 'Car', 'Bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] #اسماء الاشياء التي يوفرها سي فار
for i in range(16):
    plt.subplot(4, 4, i+1)
    #هنا قلنا له انه يسوي 16 صورة كل وحدة عبارة عن 4*4
    plt.xticks([])
    plt.yticks([])
    #هنا قلنا له باختصار يشيل الارقام عن من على المحاور
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
#plt.show()
# يمكنك استخدام كود ليعرض كل فئة مرة واحدة كأول صورة يستخرجها ومجموع 10 صور من 10 فئات من خلال هذا الكود
#import numpy as np
# for i in range(10):
    # إيجاد أول صورة من كل فئة
   # idx = np.where(training_labels == i)[0][0]
    #plt.subplot(2, 5, i+1)
    #plt.imshow(training_images[idx])
    #plt.title(class_names[i])
    #plt.axis('off')
# plt.show()
#------------------------------------------------------------------------------------------------------------
#تقدر تستخدم هذا الكود لتقليل صور سي فار من 50 الف الى 20 الف وصور الاختبار من 10الاف الى 4 الاف
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]
#------------------------------------------------------------------------------------------------------------


model = keras.models.load_model('image_classifier.keras')

img = cv2.imread('plane.jpg')
if img is None:
    print("خطأ: لم يتم العثور على الصورة!")
    exit()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (32, 32))
plt.imshow(img_resized, cmap=plt.cm.binary)
plt.axis('off')

img_input = img_resized / 255.0  # تطبيع الألوان
img_input = np.expand_dims(img_input, axis=0)  # إضافة بُعد الدفعة (يصبح الشكل: (1, 32, 32, 3)
prediction = model.predict(img_input)
index = np.argmax(prediction)

print(f"Predicted is : {class_names[index]}")
plt.show()


