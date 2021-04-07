import os
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from normalize import Normalize
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
import logging
logging.basicConfig(level=logging.DEBUG, filename='logfile_prediction.txt')

logging.debug('')
logging.debug('Model Prediction - Phase')
logging.debug('------------------------')
logging.debug('')

class CustomModelPrediction(object):
    def __init__(self, model, tokenizer, tag_encoder, X_test, y_test ):
        self._model = model
        self._tokenizer = tokenizer
        self._tag_encoder = tag_encoder
        self._X_test = X_test
        self._y_test = y_test
        #self._preprocessor = preprocessor

    def _postprocess(self, predictions):
        with open('models/unique_tags.pickle', 'rb') as handle:
            labels = pickle.load(handle)
        logging.debug("Labels type: ", type(labels))
        label_indexes = [np.argmax(prediction) for prediction in predictions]
        return [labels[label_index] for label_index in label_indexes]


    def predict(self, test_input, input_type, test_case_count=25):
        normalize = Normalize()
        if input_type == 'RANDOM_INPUT':
            input_count = 0
            for question in test_input:
                input_count += 1
                question_ = normalize.normalize(question)
                logging.debug('Test Case No.{}: {}'.format(input_count, str(question)))
                logging.debug('-' * (len(question)+16))
                logging.debug('Predicted Tags: {}'.format(self.tag_predictor(question_)))
            logging.debug('')
        else:
            test_idx = np.random.randint(len(test_input), size=test_case_count)
            logging.debug("Predicted Vs Ground Truth for {} sample".format(test_case_count))
            logging.debug('-' * 50)
            logging.debug('')
            input_count = 0
            for idx in test_idx:
                input_count += 1
                test_case = idx
                question = str(X_test[test_case])
                logging.debug('Test Case No.{}: {}'.format(input_count, question))
                logging.debug('-' * 100)
                logging.debug("Question ID:    {}".format(test_case))
                logging.debug('Predicted: ' + str(self.tag_predictor(normalize.normalize_(X_test[test_case]))))
                logging.debug('Ground Truth: ' + str(self._tag_encoder.inverse_transform(np.array([y_test[test_case]]))))
                logging.debug('\n')


    def tag_predictor(self, text):
        # Tokenize text
        X_test = pad_sequences(self._tokenizer.texts_to_sequences([text]), maxlen=400)
        # Predict
        prediction = self._model.predict([X_test])[0]
        for i, value in enumerate(prediction):
            if value > 0.5:
                prediction[i] = 1
            else:
                prediction[i] = 0
        tags = self._tag_encoder.inverse_transform(np.array([prediction]))
        return tags

    def evaluate(self, X_test, y_test, batch_size, MAX_SEQUENCE_LENGTH=400):
        score =  self._model.evaluate(X_test, y_test, batch_size=batch_size)
        # labels = self._postprocess(predictions)
        loss = score[0]
        accuracy = score[1]
        return loss, accuracy

    @classmethod
    def from_path(cls):
        import keras
        model = load_model('models/tag_predictor_keras_model.h5')

        with open('models/X_test.pickle', 'rb') as handle:
            X_test = pickle.load(handle)

        with open('models/X_test_raw.pickle', 'rb') as handle:
            X_test_raw = pickle.load(handle)

        with open('models/y_test.pickle', 'rb') as handle:
            y_test = pickle.load(handle)
        y_test = y_test.astype(float)

        with open('models/tag_encoder.pickle', 'rb') as handle:
            tag_encoder = pickle.load(handle)

        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        #model = keras.models.load_model(os.path.join(model_dir, 'tag_predictor_keras_model.h5'))
        # with open(os.path.join(model_dir, 'my_processor_state.pkl'), 'rb') as f:
        #with open('models/preprocessor.pkl', 'rb') as handle:
        #    preprocessor = pickle.load(handle)

        logging.debug(' ')
        return cls(model, tokenizer, tag_encoder, X_test, y_test)




model = CustomModelPrediction.from_path()

stackoverflow_dataset_test_input = True

if stackoverflow_dataset_test_input:
    test_input_type = 'STACKOVERFLOW'
    random_test_input = False
    #with open('models/X_test.pickle', 'rb') as handle:
        #X_test = pickle.load(handle)

    with open('models/X_test_raw.pickle', 'rb') as handle:
        X_test = pickle.load(handle)

    with open('models/y_test.pickle', 'rb') as handle:
        y_test = pickle.load(handle)
        y_test = y_test.astype(float)

    with open('models/X_test_padded.pickle', 'rb') as handle:
        X_test_padded = pickle.load(handle)

    prediction = model.predict(X_test, test_input_type)
    batch_size = 128
    loss, accuracy = model.evaluate(X_test_padded, y_test, batch_size)
    #accuracy_score_ = accuracy_score(y_test,prediction)
    #print("accuracy_score: ",accuracy_score)
    logging.debug("")
    logging.debug("Loss: {:.4f}%, Accuracy {:.2f}%".format(loss*100, accuracy * 100))

else:
    test_input_type = 'RANDOM_INPUT'
    random_test_input = True
    X_test = ['List comprehension issues in Python',
              'How to add css based colors to the fonts in html page',
              'How to fix divide by zero exception and null exception in Java?',
              'Terminal Browser trigger WEB HTML input onchange',
              "we're trying to do something in our project. We're using C# for our web application and be able to use an external device for it. But we have some small problem, we're trying to send the value from",
              "How to create custom chat icon using css only I want to create chat icon using css only. Here's a sample of it: I don't want to use bootstrap chat icons because it's hard to customize especially the numbers inside the icon grows."]
    prediction = model.predict(X_test, test_input_type)





