# Alternative_Topic-Modeling_Methods

---

## 1

----

## 2 
- d) Se puede observar que al ejecutar el programa varias veces para `TokenVieuxM.txt` desde 5 hasta 20 temas esperados, obtenemos datos como:
  
  ![Image](./coherence_values_per_nb_m_image.png)

  Podemos percatarnos que la coherencia esta en su mayor valor para la cantidad de temas esperados igual a 18 y la perplexity está en su valor mas bajo que es generalmente lo buscado. Por lo que podemos conluir que dentro del rango buscado, la mejor cantidad de temas esperados es 18.
  
- e) Se puede observar que al ejecutar el programa usando `TokenVieuxN.txt` para 10 temas esperados, sin remover las _stopwords_ tenemos una coherencia aproximada de `0.3144` mientras que para removiendo las _stopwords_ obtenemos una coherencia de `0.5781` aproximadamente, lo que indica que eliminar las _stopwords_ siempre mejora la coherencia del modelo.

- f) A pesar de que también tiende a crecer la coherencia de acuerdo a la cantidad de temas esperados, se puede observar que varía diferente en `TokenVieuxN.txt`, por lo que se puede concluir que la cantidad de temas óptimos es en función a los datos.

  ![Image](./coherence_values_per_nb_n_image.png)


----

## 3