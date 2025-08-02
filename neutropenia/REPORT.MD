# 📊 Relatório: Análise de Neutropenia  
**Disciplina:** Ciência e Visualização de Dados na Saúde – MO413  
**Instituição:** Unicamp – Instituto de Computação  
**Professor:** André Santanchè  
**Aluno:** Maurício Pereira Lopes – RA 225242  

---

## 🧬 1. Introdução

Este relatório descreve as análises realizadas sobre pacientes com neutropenia, com base em dados extraídos da base **MIMIC**, incluindo informações de admissões, diagnósticos e exames laboratoriais.

---

## 📁 2. Conjunto de Dados

Arquivos utilizados:
- `neutro_admissions.csv`
- `neutro_diagnoses.csv`
- `neutro_itemid_loinc.csv`
- `neutro_labevents_min.csv`

Cada tabela apresenta diferentes dimensões dos dados clínicos. 

---

## 📈 2.1 Análise Descritiva

### 🔢 Estatísticas Gerais

| Métrica                         | Valor     |
|-------------------------------|-----------|
| Total de admissões            | 4249      |
| Total de pacientes            | 2691      |
| Pacientes com câncer          | 1492 (55.4%) |
| Admissões com desfecho de morte | 292       |
| Mortes com diagnóstico de câncer | 99 (33.9%) |

### 🎨 Raça dos Pacientes

- Foram identificadas **2735 ocorrências de raça** para 2691 pacientes, devido à inconsistência dos dados.
- **65.7%** dos pacientes foram categorizados como **WHITE**.

### 🛏️ Dias de Internação

Gráfico (conceitual):

```
Dia de Internação | Contagem
------------------|---------
1                 | Alta
3                 | Mais frequente
6                 | Alta (especialmente em mortes)
```

### 🆘 Tipo de Admissão

| Tipo de Admissão           | Total | Óbitos | %
|---------------------------|--------|--------|----|
| EM EMER., DIRECT EMER., URGENT | 1456   | 194    | 66.4% |

### 🧾 Diagnósticos em Casos de Óbito

- Diagnósticos repetidos dificultam conclusões claras.
- Os **20 mais frequentes** foram levantados com alta sobreposição por paciente.

### 🔬 Exames em Casos de Óbito

- Análise limitada por ausência de vínculo entre diagnóstico e exames.
- Exames repetidos por paciente dificultam inferência de impacto clínico.

---

## 🔗 2.2 Análise de Correlação

Ferramenta: `Dython`

### 💡 Principais Correlacionamentos

| Variáveis                     | Correlação |
|------------------------------|------------|
| Dias de internação vs. óbito | 23%        |
| Diagnóstico vs. óbito        | 37%        |
| Diagnóstico vs. câncer       | 39–41%     |

---

## 🔮 2.3 Análise Preditiva

### 📉 Desbalanceamento

- Somente **7%** dos dados representam casos com desfecho de morte.
- Modelos enfrentaram dificuldades para prever corretamente essa classe minoritária.

### ⚙️ Estratégia de Modelagem

- Dados unificados em uma única tabela com variáveis binárias.
- Seleção da última admissão por paciente.
- Utilização da **primeira ocorrência de cada exame** para refletir estado clínico inicial.

### 🧪 Modelos Testados

#### 2.3.1 Regressão Logística

| Métrica       | Valor  |
|---------------|--------|
| Acurácia      | 58.4%  |
| F1 Score      | 11.5%  |
| Falsos Negativos | 480 |

#### 2.3.2 Random Forest

| Métrica       | Valor  |
|---------------|--------|
| Acurácia      | 93.7%  |
| F1 Score      | 0.0    |
| Verdadeiros Positivos | 0 |

#### 2.3.3 Pacientes com Câncer

Repetição das análises anteriores com os 1476 pacientes com câncer:

| Modelo        | Acurácia | F1 Score |
|---------------|----------|----------|
| Regressão Logística | 67.0%    | 13.1%     |
| Random Forest       | 91.4%    | 0.0       |

#### 2.3.4 ADASYN (Dados Sintéticos)

- Dados balanceados com ADASYN + divisão treino/validação/teste.
- Resultado promissor no treinamento, mas não generalizou:

| Modelo        | F1 Score Treino | F1 Score Teste |
|---------------|------------------|-----------------|
| Random Forest | 97.7%            | 0.0             |

---

## 📌 3. Conclusão

- Os dados têm alta variabilidade e complexidade.
- Diagnósticos e exames são excessivamente repetitivos.
- Análises preditivas falharam em identificar desfechos de óbito devido ao desbalanceamento.
- Melhor seleção de variáveis clínicas na admissão poderia levar a modelos preditivos mais eficazes.

---
