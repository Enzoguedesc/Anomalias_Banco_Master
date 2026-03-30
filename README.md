# 🔍 Análise Forense com IA no Caso Banco Master

## 📰 Resumo do Projeto
Este repositório contém uma investigação de **Análise Forense Quantitativa** aplicada ao mercado de FIDCs (Fundos de Investimento em Direitos Creditórios). O objetivo central foi validar se um pipeline de **Machine Learning** poderia ter detectado anomalias sistemáticas nos grupos **Master, Trustee, Reag e Letsbank** meses antes das intervenções do Banco Central do Brasil.

O diferencial deste projeto é a entrega: os resultados são apresentados através de um Web App em **Streamlit** com estética de **Jornal Investigativo Moderno**, convertendo métricas estatísticas em uma narrativa acessível para auditoria e tomada de decisão.

---

## 🎯 O Problema Investigado
Entre novembro de 2025 e fevereiro de 2026, oito instituições ligadas ao Grupo Master sofreram liquidação extrajudicial. O foco da investigação foi analisar:
1. **Conflitos de Interesse:** Cedentes e administradores pertencentes ao mesmo grupo econômico.
2. **Ocultação de Risco:** Uso de altas taxas de giro de carteira (aquisição de novos títulos) para mascarar ativos inadimplentes.
3. **Pontos Cegos Regulatórios:** Ausência sistemática de declaração de CNPJs de cedentes à CVM em fundos críticos.

---

## 🛠️ Como foi feito (Metodologia)

### 1. Extração e Tratamento (Pipeline de Dados)
Foram utilizados dados públicos do **Portal de Dados Abertos da CVM**. O pipeline processou os informes mensais de março a novembro de 2025, focando em 5 indicadores-chave:
* **Inadimplência Declarada:** Relação entre atrasos e Patrimônio Líquido (PL).
* **Concentração de Cedente:** Exposição ao maior devedor da carteira.
* **Giro de Carteira (Taxa de Aquisição):** Volume de novos direitos creditórios comprados.
* **Taxa de Devolução:** Recompras feitas pelo cedente (sinal de substituição de "ativos podres").
* **Ratio Inadimplência/Giro:** Indicador de "maquiagem" de balanço.

### 2. O Algoritmo: Isolation Forest
Para detectar as fraudes, utilizei o **Isolation Forest**, um algoritmo de aprendizado não supervisionado. 
* **Por que esta técnica?** Em vez de modelar o que é "normal", o algoritmo isola anomalias. Pontos anômalos são isolados com menos divisões nas árvores de decisão.
* **Treinamento:** O modelo foi treinado apenas com o "Mercado de Referência" (fundos sem vínculo com os investigados) para estabelecer a régua de normalidade.

#### Lógica Matemática:
O score de anomalia $s$ para uma amostra $x$ é calculado como:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Onde $E(h(x))$ é a profundidade média para isolar o ponto e $c(n)$ é a média de profundidade para uma árvore com $n$ nós. No projeto, scores acima de **0.60** foram tratados como alertas críticos de intervenção imediata.

---

## 💡 Principais Insights Forenses
* **Antecipação do Sinal:** O algoritmo detectou scores críticos no **Banco Master S.A.** (acima de 0.65) em **Outubro de 2025**, exatamente 30 dias antes da liquidação pelo Banco Central.
* **Casos Extremos (Outliers):** O "NF Fundo" (Trustee) e o "Engelberg 41" (Reag) atingiram scores de **0.83** e **0.74**, respectivamente, figurando como os pontos mais anômalos de toda a base de dados nacional.
* **Paradoxo da Inadimplência Zero:** Detectamos fundos com giro de carteira superior a 20% ao mês e inadimplência declarada de 0%. Estatisticamente, isso provou ser um comportamento artificial, indicando a "limpeza" constante de ativos ruins através de novas emissões.

---

## 💻 Stack Tecnológica
* **Linguagem:** Python
* **Machine Learning:** `Scikit-learn` (Isolation Forest, StandardScaler)
* **Processamento:** `Pandas`, `NumPy`
* **Visualização:** `Plotly` (Gráficos interativos e Violin Plots para análise de densidade)
* **Interface:** `Streamlit` (Layout Jornalístico com CSS Customizado)

---


> **Disclaimer:** Este projeto tem finalidade estritamente acadêmica e investigativa. Todas as análises baseiam-se em dados públicos da CVM e metodologias estatísticas reprodutíveis.
