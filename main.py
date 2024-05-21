from flask import Flask, render_template, url_for, flash, redirect, request,jsonify
from flask_cors import CORS
# from forms import RegistrationForm, LoginForm
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import shutil
from tensorflow import keras
from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import DenseNet121
# from keras import layers
# from keras import Model
# from keras.optimizers import Adam
from keras.preprocessing import image
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model = load_model('Densenetmodel.h5')    
# class names
class_names=np.load('./class_names.npy')


# Load the fine-tuned model and tokenizer
# model_path = "./fine_tuned_pegasus_model"
# summarizationmodel = PegasusForConditionalGeneration.from_pretrained(model_path)
# tokenizer = PegasusTokenizer.from_pretrained(model_path)

with open("herb_summaries.pkl", "rb") as file:
    loaded_summaries = pickle.load(file)


posts = [
    {
        'author': '',
        'title': '',
        'content': '',
        'date_posted': ''
    }
]

app = Flask(__name__)
CORS(app)
# app.config['SECRET_KEY'] = 'de7b2121dc14f07073c8ce6511f47422'
UPLOAD_FOLDER = './static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)
@app.route("/test")
def test():
    return "This a test"
@app.route("/upload", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    for file_name in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    file = request.files['image']
    print("File received")
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to the UPLOAD_FOLDER
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

    # json_data = request.get_json()

    # if 'imagePath' in json_data:
    # image_path = json_data['imagePath']
    # return image_path
        # return jsonify({'res': 'getting_data'}), 200
    
    # path = "./static/images/"
    # shutil.rmtree(path)
    # path = os.path.join("./", path)
    # os.mkdir(path)
    # imagefile= request.files['imagefile']
    # # print(imagefile)
    # image_path = os.path.join(path,imagefile.filename)

    # imagefile.save(image_path)
    
    # # Return the dynamically generated filename
    test_img_path = './static/images/'+os.listdir('./static/images/')[0]
    
    output = image_segmentation_and_classification(test_img_path)
    print(output)
    # # summary = text_summarization(output)
    # print(image_path.replace('./static',''))
    # return render_template('HerbIdentifier.html', Classified_image=output, image_path=image_path.replace('./static',''),summary=loaded_summaries[output])

    # return jsonify(output, image_path.replace('./static',''), loaded_summaries[output])
    if(output == "Unknown"):
        return jsonify({'classlabel' : output, 'summary': ""})
    
    for file_name in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    return jsonify({'classlabel' : output, 'summary' : loaded_summaries[output]})

def image_segmentation_and_classification(test_img_path):
    
    #image path
    # test_img_path = './static/images/'+os.listdir('./static/images/')[0]
    # print(test_img_path)

    main_img = cv2.imread(test_img_path)
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(img, (1600, 1200))

    gs = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gs, (55,55),0)

    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((50,50),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    def find_contour(cnts):
        contains = []
        y_ri,x_ri, _ = resized_image.shape
        for cc in cnts:
            yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
            contains.append(yn)
        val = [contains.index(temp) for temp in contains if temp>0]
        return val[0]
    
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        
    black_img = np.empty([1200,1600,3],dtype=np.uint8)
    black_img.fill(0)

    index = find_contour(contours)
    cnt = contours[index]
    mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)

    maskedImg = cv2.bitwise_and(resized_image, mask)
    white_pix = [255,255,255]
    black_pix = [0,0,0]

    final_img = maskedImg
    h,w,channels = final_img.shape
    for x in range(0,w):
        for y in range(0,h):
            channels_xy = final_img[y,x]
            if all(channels_xy == black_pix):
                final_img[y,x] = white_pix
    
    

    # Assuming 'final_img' is your numpy image variable
    test_image = final_img

    # Resize the test image to match the expected input shape of the model
    resized_test_image = cv2.resize(test_image, (224, 224))

    # Preprocess the test image for prediction
    preprocessed_test_image = image.img_to_array(resized_test_image)
    preprocessed_test_image = np.expand_dims(preprocessed_test_image, axis=0)
    preprocessed_test_image = preprocessed_test_image / 255.0  # Normalize

    threshold = 0.8495697617530823
    # Make predictions using the trained model
    predictions = model.predict(preprocessed_test_image)
    max_prob = np.max(predictions)
    # Check if the maximum probability is above the threshold
    if max_prob < threshold:
        return "Unknown"
    # Display a message on Herb Identifier Page that image is an outlier.
    # Get the predicted class label

    else:
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_names[predicted_class_index]
        # Return the predicted class label
        return predicted_class_label

# def text_summarization(predicted_class_label):

    # Path to the .txt file containing the sample document
    file_path = "./Knowledge_Base/"+ predicted_class_label + ".txt"

    # Read the content of the .txt file
    with open(file_path, 'r', encoding='utf-8') as file:
        input_document = file.read()

    # Tokenize the input document
    input_ids = tokenizer(input_document, return_tensors="pt").input_ids

    # Generate the summary
    summary_ids = summarizationmodel.generate(input_ids, max_length=1024,min_length=60, num_beams=4, length_penalty=2.0, early_stopping=True)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return generated_summary

@app.route("/about")
def about():
    return render_template('about.html', title='About')

# @app.route("/register", methods=['GET', 'POST'])
# def register():
#     form = RegistrationForm()
#     if form.validate_on_submit():
#         flash(f'Account created for {form.username.data}!', 'success')
#         return redirect(url_for('home'))
#     return render_template('register.html', title='Register', form=form)

# @app.route("/login", methods=['GET', 'POST'])
# def login():
#     form = LoginForm()
#     if form.validate_on_submit():
#         if form.email.data == 'admin@blog.com' and form.password.data == 'password':
#             flash('You have been logged in!', 'success')
#             return redirect(url_for('home'))
#         else:
#             flash('Login Unsuccessful. Please check username and password','danger')
#     return render_template('login.html', title='Login', form=form) 

if __name__=='__main__':
    app.run(debug=True, port=5001)