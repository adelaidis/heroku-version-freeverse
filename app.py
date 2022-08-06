import os
from flask import Flask, render_template, request
from wtforms import StringField
from wtforms import Form, StringField, validators, SubmitField
from appUtils import *
from tensorflow.python.framework import ops
ops.get_default_graph()

# create app
app = Flask(__name__)

class Form(Form):
    seed = StringField("Enter the title of the poem", validators=[
                     validators.InputRequired()])
    submit = SubmitField("Submit")


# Home page
# if user goes to /, they're on the main page, so show them the home page
@app.route("/", methods=['GET', 'POST'])
def index():
    form = Form(request.form)
    # If submit clicked
    if request.method == "POST" and form.validate():   
        # take seed field from form
        seed = request.form['seed']
        # load model
        model = load_keras_model()
        # tokenize data
        tokenizer = prepare_tokenization()
        # and render the page again with poem displayed
        return render_template('index.html', form = form, seed = seed, prediction_text = create_poem(model, tokenizer, seed))  
    else:
        # Send template information to index.html
        return render_template('index.html', form=form)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="localhost", port=5000, debug=True)

    