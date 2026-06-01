# Sistema Híbrido de Recomendação de Filmes: Vetores, Espaço Vetorial e Avaliação Social

Este repositório contém o desenvolvimento de um **Sistema de Recomendação de Filmes Híbrido**, construído em Python. O projeto original utilizava metadados textuais compostos por sinopse, palavras-chave, gêneros, elenco (`cast`) e equipe técnica (`crew`) para calcular a proximidade entre os títulos usando álgebra linear (Espaço Vetorial e Similaridade de Cosseno).

Nesta versão customizada, o algoritmo foi aprimorado com foco em duas modificações estratégicas:
1. **Foco Temático Puro:** Exclusão das colunas `cast` e `crew` para garantir que o motor de recomendação analise estritamente a essência da história, enredo e gênero, sem influência de atores ou diretores.
2. **Métrica de Decisão Social (Rating):** Integração da coluna numérica `vote_average` (avaliação do filme), permitindo que o sistema ordene as recomendações ponderando a proximidade temática e a aprovação do público.

---

## 🚀 Arquitetura e Pipeline de Dados

O projeto combina processamento de linguagem natural (PLN) para o processamento de texto e técnicas de normalização estatística para os dados numéricos.

### 1. Processamento de Texto (NLP)
* **Consolidação de Tags:** Os campos textuais `genres`, `keywords` e `overview` são limpos, transformados em listas e concatenados em uma única string agregadora chamada `tags`.
* **Stemming Textual:** Utilização do algoritmo `PorterStemmer` da biblioteca `NLTK` para reduzir as palavras aos seus radicais morfológicos (ex: *action*, *actions* e *actioned* são reduzidos para o mesmo radical), otimizando a frequência de termos.
* **Vetorização (Bag of Words):** Uso do `CountVectorizer` (Scikit-Learn) configurado para selecionar as 5.000 palavras mais frequentes, desconsiderando *stop words* em inglês. Isso converte o texto em um espaço vetorial multidimensional.

### 2. Engenharia de Features Numéricas
Como a similaridade de cosseno resulta em um valor contido entre `0.0` e `1.0`, a nota média do filme (`vote_average`), que originalmente varia na escala de `0.0` a `10.0`, passaria a esmagar o cálculo de distância textual se usada sem tratamento.
* **Normalização Linear:** Aplicação do `MinMaxScaler` do Scikit-Learn para reescalonar a variável `vote_average` estritamente para o intervalo `[0, 1]`, permitindo uma combinação matemática justa e equilibrada.

---

## 🧠 Algoritmo de Recomendação Híbrido

O motor de cálculo final utiliza uma abordagem ponderada linear para unir os dois mundos (texto e número). A pontuação final de relevância para cada par de filmes é determinada pela fórmula:

$$\text{Score Final} = (S_{\text{cosseno}} \times W_{\text{similaridade}}) + (N_{\text{norm}} \times W_{\text{nota}})$$

Onde:
* **$S_{\text{cosseno}}$**: Coeficiente de similaridade de cosseno gerado através do cruzamento angular das matrizes de texto.
* **$N_{\text{norm}}$**: Valor normalizado da nota média do filme (entre 0.0 e 1.0).
* **$W_{\text{similaridade}}$**: Peso atribuído à proximidade do enredo (Configuração padrão: `0.8`).
* **$W_{\text{nota}}$**: Peso atribuído à avaliação do público (Configuração padrão: `0.2`).

---

## 🛠️ Implementação do Código

Abaixo está o bloco principal da função que realiza a busca e ordenação dos resultados de forma dinâmica:

```python
def sistema_recomendacao(movie, peso_similaridade=0.8, peso_nota=0.2):
    # Obtém o índice absoluto do filme de busca no DataFrame
    index = df_dsa_filmes_novo[df_dsa_filmes_novo['title'] == movie].index[0]
    
    # Extrai o vetor de similaridade de cosseno para o filme selecionado
    sim_scores = similaridades[index]
    
    pontuacoes_finais = []
    
    for i, sim_score in enumerate(sim_scores):
        # Evita recomendar o próprio filme de busca
        if i != index: 
            # Captura a nota normalizada do filme comparado
            nota_norm = df_dsa_filmes_novo.iloc[i]['vote_average_norm']
            
            # Executa a Combinação Linear Ponderada
            score_final = (sim_score * peso_similaridade) + (nota_norm * peso_nota)
            
            pontuacoes_finais.append((i, score_final))
            
    # Ordena os filmes de forma decrescente com base no Score Híbrido
    pontuacoes_finais = sorted(pontuacoes_finais, reverse=True, key=lambda x: x[1])
    
    # Exibe o Top 5 das recomendações refinadas
    print(f"Recomendações baseadas em '{movie}':\n")
    for i in pontuacoes_finais[0:5]:
        filme_recommended = df_dsa_filmes_novo.iloc[i[0]].title
        score_obtido = i[1]
        print(f"- {filme_recommended} (Score Híbrido: {score_obtido:.4f})")
```

---

## 📊 Exemplo de Execução

Ao pesquisar recomendações para o filme **"The Hobbit: The Battle of the Five Armies"**, o modelo ponderado apresenta títulos altamente correlacionados em termos de enredo mitológico/fantasia, ordenando-os de forma inteligente com o viés de avaliação:

```text
Recomendações baseadas em 'The Hobbit: The Battle of the Five Armies':

- The Hobbit: An Unexpected Journey (Score Híbrido: 0.5053)
- The Hobbit: The Desolation of Smaug (Score Híbrido: 0.4904)
- The Lord of the Rings: The Fellowship of the Ring (Score Híbrido: 0.4357)
- The Lord of the Rings: The Two Towers (Score Híbrido: 0.4177)
- The Lord of the Rings: The Return of the King (Score Híbrido: 0.3952)
```

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Python 3.x**
* **Pandas & NumPy:** Manipulação, limpeza e estruturação de dados matriciais.
* **Scikit-Learn:** Engenharia de atributos (`MinMaxScaler`, `CountVectorizer`) e cálculo de distância angular (`cosine_similarity`).
* **NLTK (Natural Language Toolkit):** Processamento de linguagem natural e aplicação de Stemming (`PorterStemmer`).

---
*Projeto desenvolvido como estudo prático de álgebra linear e processamento de linguagem natural aplicados a problemas reais de inteligência de dados.*
