#!/usr/bin/python
# -*- coding: utf-8 -*-

import io
import math as m
import re
import string

import en_core_web_sm
import es_core_news_sm
import fr_core_news_sm
import matplotlib.pyplot as plt
import matplotx
import numpy as np
import pandas as pd
import razdel
import requests
import seaborn as sns
import spacy
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pymystem3 import Mystem  # lemmatizing from yandex
from scipy import spatial
from sentence_transformers import SentenceTransformer

## Предобработка текста

nltk.download('punkt')
spec_chars = string.punctuation + '\n\xa0«»\t—'


def extract_text_from_pdf(fh):
    """
    Функция считывающая текст из pdf-файла в строку python
    
    Ввод: путь до файла на компьютере
    Вывод: строка python
    """

    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    for page in PDFPage.get_pages(fh, caching=True,
                                  check_extractable=True):
        page_interpreter.process_page(page)

    text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()

    if text:
        return text


def remove_chars_from_text(text, chars):
    return ''.join([ch for ch in text if ch not in chars])


def count_syllables(text, lang):
    if lang == 'rus':
        literas = 'аоиыуэ'
    elif lang == 'spa':
        literas = 'aеiоuáíóúé'
    elif lang == 'eng':
        literas = 'aeiou'
    elif lang == 'fra':
        literas = 'aeiouáíóúé'
    else:
        raise ValueError

    count = 0
    for i in text:
        if i in literas:
            count += 1
    return count


def del_stopwords(lang, text_tokens):
    if lang == 'rus':
        lang_stopwords = stopwords.words('russian')
    elif lang == 'spa':
        lang_stopwords = stopwords.words('spanish')
    elif lang == 'eng':
        lang_stopwords = stopwords.words('english')
    elif lang == 'fra':
        lang_stopwords = stopwords.words('french')
    else:
        raise ValueError

    filtered_tokens = []

    for token in text_tokens:
        if token not in lang_stopwords:
            filtered_tokens.append(token)

    return filtered_tokens


def lemmatize(lang, text):
    if lang == 'rus':
        m = Mystem()
        lemmas = m.lemmatize(text)
        lemmatize_str = ''.join(lemmas).strip()
    elif lang == 'spa':
        nlp = es_core_news_sm.load()
        list = []
        for token in nlp(text):
            list.append(token.lemma_)
        lemmatize_str = ' '.join(list)
    elif lang == 'eng':
        nlp = en_core_web_sm.load()
        list = []
        for token in nlp(text):
            list.append(token.lemma_)
        lemmatize_str = ' '.join(list)
    elif lang == 'fra':
        nlp = fr_core_news_sm.load()
        list = []
        for token in nlp(text):
            list.append(token.lemma_)
        lemmatize_str = ' '.join(list)
    return lemmatize_str


def text_preprocessing(path, lang):

    text = extract_text_from_pdf(path)  # Прочитали текст из PDF
    sent_text = list(x.text for x in razdel.sentenize(text))  # Разделили строку на массив строк - предложений

    num_of_sent = len(sent_text)
    print ('Number of sentences: ', num_of_sent)

    text = remove_chars_from_text(text, spec_chars)  # Удалили из текста специальные символы
    text_without_spaces = remove_chars_from_text(text, ' ')  # Удалили из текста пробелы
    num_of_symbols = len(text_without_spaces)
    print ('Number of symbols: ', num_of_symbols)

    text = text.lower()  # Привели текст к нижнему регистру

    num_of_syllables = count_syllables(text_without_spaces, lang)  # Количество слогов во всем тексте
    print ('Number of syllables: ', num_of_syllables)

    text_tokens = word_tokenize(text)  # Разделили строку на массив строк - слов
    count_tokens_with_stopwords = len(text_tokens)  # Количество слов во всем тексте
    print ('Number of tokens with stopword: ',
           count_tokens_with_stopwords)

    count_words_w_3_syllables = 0
    for word in text_tokens:
        w_cnt_syl_3 = count_syllables(str(word), lang)
        if w_cnt_syl_3 >= 3:
            count_words_w_3_syllables += 1

    print ('Number of tokens with 3 syllables: ',
           count_words_w_3_syllables)  # Количество слов с 3 слогами или больше

    filtered_tokens = del_stopwords(lang, text_tokens)  # Удалили из массива слов стоп-слова
    count_tokens = len(filtered_tokens)
    print ('Number of tokens without stopword: ', count_tokens)

    filtered_string = ' '.join(filtered_tokens)  # Преобразовали список в строку

    lemmatize_str = lemmatize(lang, filtered_string)  # Привели слова к леммам
    lemmatize_tokens = lemmatize_str.split()

    unique_tokens = list(set(lemmatize_tokens))
    unique_tokens_with_stopwords = list(set(text_tokens))
    count_unique_tokens_with_stopwords = \
        len(unique_tokens_with_stopwords)
    count_unique_tokens = len(unique_tokens)
    print ('Number of unique tokens without stopword: ',
           count_unique_tokens)
    print ('Number of unique tokens with stopword: ',
           count_unique_tokens_with_stopwords)

    res = {
        'num_of_sent': num_of_sent,
        'num_of_symbols': num_of_symbols,
        'num_of_syllables': num_of_syllables,
        'count_tokens_with_stopwords': count_tokens_with_stopwords,
        'count_words_w_3_syllables': count_words_w_3_syllables,
        'count_tokens': count_tokens,
        'count_unique_tokens': count_unique_tokens,
        'count_unique_tokens_with_stopwords': count_unique_tokens_with_stopwords,
        'lemmatize_tokens': lemmatize_tokens,
        'unique_tokens': unique_tokens,
        'text_tokens': text_tokens,
    }

    return res


def batch(iterable, n=50):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def do_dict_freq(unique_tokens, lang, year):
    inf_unique_tokens = [x + '_INF' for x in unique_tokens]
    list_batches = batch(inf_unique_tokens)
    num_batches = (len(list(list_batches)))
    list_batches = batch(inf_unique_tokens)
    cnt = 0
    dict_freq = {}
    for item in list_batches:
        cnt += 1
        print('Batch', cnt, 'of',  num_batches)
        inf_unique_tokens_str = ','.join(item)

        if lang == 'rus':
            params = {
                "content": inf_unique_tokens_str,
                "year_start": str(year - 1),
                "year_end": str(year),
                "corpus": "ru-2019"
            }
        elif lang == 'spa':
            params = {
                "content": inf_unique_tokens_str,
                "year_start": str(year - 1),
                "year_end": str(year),
                "corpus": "es-2019"
            }
        elif lang == 'eng':
            params = {
                "content": inf_unique_tokens_str,
                "year_start": str(year - 1),
                "year_end": str(year),
                "corpus": "en-2019"
            }
        elif lang == 'fra':
            params = {
                "content": inf_unique_tokens_str,
                "year_start": str(year - 1),
                "year_end": str(year),
                "corpus": "fr-2019"
            }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.125 Safari/537.36",
        }

        r = requests.get("https://books.google.com/ngrams/json", params=params, headers=headers, timeout=1000)
        html = r.text
        time_series = pd.read_json(html, typ="series")

        for i in time_series:
            if i['type'] == 'EXPANSION' and len(i['timeseries']) == 2:
                if i['parent'][:int(i['parent'].find("_"))] not in dict_freq:
                    dict_freq[i['parent'][:int(i['parent'].find("_"))]] = [i['timeseries'][1]*100, 1]
                else:
                    dict_freq[i['parent'][:int(i['parent'].find("_"))]][0] = dict_freq[i['parent'][:int(i['parent'].find("_"))]][0] \
                                                                            + i['timeseries'][1]*100
                    dict_freq[i['parent'][:int(i['parent'].find("_"))]][1] = dict_freq[i['parent'][:int(i['parent'].find("_"))]][1] + 1


    return dict_freq



def main(file, lang, year):
    res = text_preprocessing(file, lang)
    print()

    lemmatize_tokens_dict = {}
    for word in res['lemmatize_tokens']:
        if word in lemmatize_tokens_dict:
            lemmatize_tokens_dict[word] = lemmatize_tokens_dict[word] + 1
        else:
            lemmatize_tokens_dict[word] = 1

    dict_freq = do_dict_freq(res['unique_tokens'], lang, year)

    words_calculated_freq = {}
    for i in dict_freq:
        words_calculated_freq[i] = dict_freq[i][0] / dict_freq[i][1]

    def most_least_common(words_calculated_freq, n=10):
        sorted_freq = dict(sorted(words_calculated_freq.items(), key=lambda item: item[1]))
        from collections import Counter
        d = Counter(words_calculated_freq)


        sum1 = 0
        for k, v in d.most_common(n):
            sum1 += v
            
        most_common = d.most_common(n)
        least_common = d.most_common()[:-n-1:-1]
        
        return most_common, least_common

    most_common, least_common = most_least_common(words_calculated_freq)

    df_most_common = pd.DataFrame(most_common, columns=['Word',
                                  'Frequency'])
    df_least_common = pd.DataFrame(least_common, columns=['Word',
                                   'Frequency'])

    print("\nMost common words")
    print(df_most_common)
    print("\nLeast common words")
    print(df_least_common)

    sum = 0
    n = 0
    for i in words_calculated_freq:
        sum = sum + lemmatize_tokens_dict[i] * words_calculated_freq[i]
        n = n + lemmatize_tokens_dict[i]

    avg_freq = sum / n
    print('\nAverage frequency (ref. Google corpus): {:.10f} %'.format(avg_freq))

    # # Расчет статистических характеристик

    num_of_sent = res['num_of_sent']
    num_of_symbols = res['num_of_symbols']
    num_of_syllables = res['num_of_syllables']
    count_tokens_with_stopwords = res['count_tokens_with_stopwords']
    count_words_w_3_syllables = res['count_words_w_3_syllables']
    count_tokens = res['count_tokens']
    count_unique_tokens = res['count_unique_tokens']
    count_unique_tokens_with_stopwords = \
        res['count_unique_tokens_with_stopwords']
    lemmatize_tokens = res['lemmatize_tokens']
    unique_tokens = res['unique_tokens']
    text_tokens = res['text_tokens']

    print()

    Lp = count_tokens_with_stopwords / num_of_sent
    print ('Average sentence length:', round(Lp, 2))
    Lpp = num_of_symbols / count_tokens_with_stopwords
    print ('Average word length:', round(Lpp, 2))
    Pp = (count_tokens_with_stopwords - count_tokens) \
        / count_tokens_with_stopwords
    print ('Frequency of use of stop words:', round(Pp, 2))
    TTR = count_unique_tokens_with_stopwords \
        / count_tokens_with_stopwords
    print ('Lexical diversity (Type-Token Ratio):', round(TTR, 2))
    R = count_unique_tokens_with_stopwords \
        / m.sqrt(count_tokens_with_stopwords)
    print ("Lexical diversity (Guiraud's Root TTR):", round(R, 2))
    U = m.log(count_tokens_with_stopwords) ** 2 \
        / (m.log(count_tokens_with_stopwords)
           - m.log(count_unique_tokens_with_stopwords))
    print ("Lexical diversity (Dugast's Uber Index):", round(U, 2))

    result = {}
    for word in text_tokens:
        result[word] = (result[word] + 1 if word in result else 1)

    df_res = pd.DataFrame(result.items(), columns=['word', 'count'])
    a_df = df_res.groupby(['count']).count()
    a_df = a_df.reset_index()

    # display(a_df)
    # print(a_df.shape)

    k_sum = 0
    for i in range(a_df.shape[0]):
        k_sum = k_sum + int(a_df.loc[i, 'word']) * (int(a_df.loc[i,'count']) / count_tokens_with_stopwords) ** 2

    K = 10 ** 4 * (-1 / count_tokens_with_stopwords + k_sum)
    print ("Lexical diversity (Yule's K):", round(K, 2))

    FK = 0.39 * (count_tokens_with_stopwords / num_of_sent) + 11.8 \
        * (num_of_syllables / count_tokens_with_stopwords) - 15.59
    print ("Readability Index (Flesch–Kincaid):", round(FK, 2))
    G = 0.4 * (count_tokens_with_stopwords / num_of_sent + 100
               * (count_words_w_3_syllables
               / count_tokens_with_stopwords))
    print ('Readability Index (Gunning Fog):', round(G, 2))
