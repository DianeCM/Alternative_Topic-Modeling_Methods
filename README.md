# Alternative_Topic-Modeling_Methods

---

## 1

- a)  

- b)  

----

## 2

- a) Stopwords are commonly used words in a language that are considered to have little or no semantic meaning when used in a text. These words are often removed from a text before any analysis is performed, as they do not carry important information about the meaning of the text and can unnecessarily increase the processing time of the analysis. In natural language processing (NLP), stopwords are typically defined as a predefined set of words. The specific list of stopwords can vary depending on the application and the language being used, but the basic idea is to remove words that are so common and uninformative that they are unlikely to contribute to the meaning of the text. This process can be useful for tasks such as text classification, sentiment analysis, and topic modeling, where the goal is to extract meaningful information from the text while ignoring noise and irrelevant words.
- b)  

- c)  
 
- d) Se puede observar que al ejecutar el programa varias veces para `TokenVieuxM.txt` desde 5 hasta 20 temas esperados, obtenemos datos como:
  
  ![Image](./coherence_values_per_nb_m_image.png)

  Podemos percatarnos que la coherencia esta en su mayor valor para la cantidad de temas esperados igual a 18 y la perplexity está en su valor mas bajo que es generalmente lo buscado. Por lo que podemos conluir que dentro del rango buscado, la mejor cantidad de temas esperados es 18.
  
- e) Se puede observar que al ejecutar el programa usando `TokenVieuxN.txt` para 10 temas esperados, sin remover las _stopwords_ tenemos una coherencia aproximada de `0.3144` mientras que para removiendo las _stopwords_ obtenemos una coherencia de `0.5781` aproximadamente, lo que indica que eliminar las _stopwords_ siempre mejora la coherencia del modelo.

- f) A pesar de que también tiende a crecer la coherencia de acuerdo a la cantidad de temas esperados, se puede observar que varía diferente en `TokenVieuxN.txt`, por lo que se puede concluir que la cantidad de temas óptimos es en función a los datos.

  ![Image](./coherence_values_per_nb_n_image.png)

----

## 3