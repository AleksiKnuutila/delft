import json
from utilities.Embeddings import Embeddings
from utilities.Utilities import split_data_and_labels
from textClassification.reader import load_texts_and_classes_pandas
from textClassification.reader import load_texts_pandas
import textClassification
import argparse
import keras.backend as K
import pandas as pd
import time
import pdb

list_classes = [ 'Ajoneuvot ja liikenne', 'Koti ja rakentaminen', 'Paikkakunnat', 'Suhteet', 'Terveys', 'Yhteiskunta' ]


def train(embeddings_name, fold_count):
#    model = textClassification.Classifier('suomi24', "gru", list_classes=list_classes, max_epoch=30,
    model = textClassification.Classifier('suomi24', "gru", list_classes=list_classes, max_epoch=1,
        fold_number=fold_count, embeddings_name=embeddings_name)

    print('loading train dataset...')
    xtr, y = load_texts_and_classes_pandas("data/textClassification/suomi24/suomi24.csv")
    if fold_count == 1:
        model.train(xtr, y)
    else:
        model.train_nfold(xtr, y)
    # saving the model
    model.save()

def train_and_eval(embeddings_name, fold_count):
    model = textClassification.Classifier('suomi24', "gru", list_classes=list_classes, max_epoch=1, fold_number=fold_count,
        use_roc_auc=True, embeddings_name=embeddings_name)

    print('loading train dataset...')
    xtr, y = load_texts_and_classes_pandas("data/textClassification/suomi24/suomi24.csv")

    # segment train and eval sets
    x_train, y_train, x_test, y_test = split_data_and_labels(xtr, y, 0.9)

    if fold_count == 1:
        model.train(x_train, y_train)
    else:
        model.train_nfold(x_train, y_train)
    model.eval(x_test, y_test)

    # saving the model
    model.save()


def test():
    # load model
    model = textClassification.Classifier('toxic', "gru", list_classes=list_classes)
    model.load()

    print('loading test dataset...')
    xte = load_texts_pandas("data/textClassification/toxic/test.csv")
    print('number of texts to classify:', len(xte))
    start_time = time.time()
    result = model.predict(xte, output_format="csv")
    print("runtime: %s seconds " % (round(time.time() - start_time, 3)))
    return result


# classify a list of texts
def classify(texts, output_format):
    # load model
    model = textClassification.Classifier('suomi24', "gru", list_classes=list_classes)
    model.load()
    start_time = time.time()
    result = model.predict(texts, output_format)
    print("runtime: %s seconds " % (round(time.time() - start_time, 3)))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classification of comments/short texts in toxicity types (toxic, severe_toxic, obscene, threat, insult, identity_hate)")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()

    action = args.action
    if action not in ('train', 'train_eval', 'classify', 'test'):
        print('action not specifed, must be one of [train,train_eval,test,classify]')

    # change bellow for the desired pre-trained word embeddings using their descriptions in the file
    # embedding-registry.json
    # be sure to use here the same name as in the registry ('glove-840B', 'fasttext-crawl', 'word2vec'),
    # and that the path in the registry to the embedding file is correct on your system
    embeddings_name = "fasttext-finnish"

    if action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        train(embeddings_name, args.fold_count)

    if action == 'test':
        y_test = test()

        # write test predictions as a submission file
        sample_submission = pd.read_csv("data/textClassification/toxic/sample_submission.csv")
        sample_submission[list_classes] = y_test
        sample_submission.to_csv("data/textClassification/toxic/result.csv", index=False)

    if args.action == 'train_eval':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")

        y_test = train_and_eval(embeddings_name, args.fold_count)

    if action == 'classify':
        someTexts = ["Olen ollut suhteessa poikaystävä kanssa nyt vajaa vuoden. Olemme todella läheisiä ja olemme puhuneet kaikkea maan ja taivaan väliltä. En ole välimatkan takia nähnyt vielä kasvotusten hänen perheensä kanssa mutta olen jutellut hänen siskonsa kanssa ja olen tietoinen, että hänen perheensä ainakin tietää ette minä ja poikaystäväni olemme yhteyksissä (En tiedä tietävätkö suhteemme laadusta). Jokatapauksessa juttu on se, että poikaystäväni on sairas(Syöpä ja saa siihen hoitoa) ja hän on todella huonossa kunnossa sairaalassa ja päätin ilahduttaa kukilla. Kukat oli määrä toimittaa sairalaan missä hän on mutta tämä kyseinen kukkafirma soitti numeroon, jonka annoin ja poikaystäväni sisko tietysti vastasi (hän ei pysty puhumaan tällä hetkellä tilansa takia) ja sisko halusi mielummin hakea kukat läheiseltä asemalta kun saada kukat sairaalaan suoraan ja nyt jäin miettimään miksi näin olisi? ... Johtuuko siitä, että eivät halua vanhempien näkevän jonkun naisen lähettävän kukkia vai mikä on homman nimi?.... Ehkä tyhmää spekulaatiota mutta mietin mikä tässä voisi olla takana.....",
        "Kokoomuksen sanoma on tämä, että kun suomessa ei ole tarpeeksi työvoimaa, niin pitäisi saada lisää ulkomailta työntekijöitä. Eli Suomessa on muka työvoimapula. Suomeksi sanottuna pitäisi siis saada ihmisiä jotka tekevät raskaita töitä sellaisella palkalla millä ei tule Suomessa toimeen. Joutuvat tekemään jopa kahta työtä, niinkuin monissa maissa, että edes perustoimeentulo olisi turvattu. Samaan aikaan satojatuhansia suomalaisia on työttömänä. Se on se Kokoomuksen sanoma, köyhät kyykkyyn ja jos ei työ nälkäpalkalla maistu, niin otetaan ulkomailta orjatyövoimaa tilalle. Johtajille sen sijaan voidaan jakaa miljoonien tulospalkkioita, kun ovat taas onnistuneet saneeraamaan yritykset kuntoon, eli osan työntekijöistä työttömiksi.",
        "Jokainen, joka katuvaisena syntinsä tunnustaen on Jumalan edessä pyytäen Jumalalta syntejään anteeksi Jeesuksen Kristuksen ja Hänen sovitustyönsä tähden, on armahdettu lammas, kuuluu hän sitten mihin tahansa uskonnolliseen ryhmään. Me ihmiset voimme määritellä ja karsinoida lampaat milloin mihinkin kuuluviksi, mutta Jumalalle kelpaa kaikki , vain ja yksin Kristuksen veren puhdistamat lampaat, olivatpa ne minkä nimisiä tahansa. Paikka ei ole tärkeä, missä puhdistus. Et tule mainitsemaan viimeisellä tuomiolla missä (seurakunnassa) tai kenen (ihmisvälimiehen) kautta, vaan ainoastaan: Yksin ylistys ja kiitos HÄNEN PYHÄLLE NIMELLEEN!!!"]
        result = classify(someTexts, "json")
        print(json.dumps(result, sort_keys=False, indent=4, ensure_ascii=False))

    # See https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
