# 📌 Quadro Kanban do Projeto: IA para Análise Preditiva (MNIST)

Este documento registra o planejamento, a rota de desenvolvimento, o status de cada tarefa e o histórico de commits do projeto, garantindo total rastreabilidade do processo de engenharia de software e ciência de dados.

---

## 📊 Visão Geral do Tablero

| 📋 Backlog | ⏳ Em Progresso | 🔍 Em Revisão / Validação | ✅ Concluído |
| :--- | :--- | :--- | :--- |
| **Passo 4**: Fase 3 - Modelagem (SVM, RF, MLP) | ⏳ Em Progresso |
| **Passo 5**: Fase 4 - Matrizes & Benchmark | | | |
| **Passo 6**: Fase 5.1/5.2 - OOD & Máscaras | | | |
| **Passo 7**: Fase 5.3 - Imagens Próprias (CV) | | | |
| **Passo 8**: Documentação (README) & Vídeo | | | |

| **Passo 1**: Setup Inicial, Ambiente & Git | ✅ Concluído |
| **Passo 2**: Fase 1 - EDA & Carga MNIST | ✅ Concluído | 
| **Passo 3**: Fase 2 - Split & Normalização | ✅ Concluído |
---

## 🗺️ Detalhamento das Etapas e Checklist de Entregas

### [x] Passo 1: Inicialização do Repositório, Ambiente & Estrutura
- [x] Criar estrutura de diretórios (`data/custom_digits/`, `notebooks/`, `src/`).
- [x] Configurar `.gitignore` e `requirements.txt`.
- [x] Inicializar Git (`main`), primeiro commit e criação da branch `develop`.
- [x] Criar ambiente virtual `.venv` e instalar dependências.
- [x] Registrar kernel do Jupyter (`miniprojeto-m2`).
- [x] Criar arquivo `KANBAN.md` de acompanhamento.
- **Branch**: `main` -> `develop`
- **Commit**: `configura estrutura inicial do projeto e dependencias`

---

### [x] Passo 2: Fase 1 - Carregamento e Análise Exploratória de Imagens (EDA)
- [x] Criar branch `feature/fase1-eda` a partir de `develop`.
- [x] Criar notebook `notebooks/mnist_pipeline.ipynb` com cabeçalhos estruturados.
- [x] Carregar dataset MNIST (`fetch_openml('mnist_784')`).
- [x] Analisar dimensões das matrizes $X$ (70.000, 784) e $y$ (70.000).
- [x] Verificar e comprovar o balanceamento de classes (dígitos de 0 a 9) com gráfico e tabela.
- [x] Construir grade visual ($2 \times 5$) com `matplotlib` exibindo amostras reais de cada dígito com seus rótulos.
- [x] Célula conceitual: Explicar detalhadamente a escala de intensidade de pixels (0 a 255) e a representação vetorial ($28 \times 28 = 784$ features).
- **Branch**: `feature/fase1-eda`
- **Commit Planejado**: `implementa analise exploratoria e carregamento do mnist`

---

### [ ] Passo 3: Fase 2 - Pipeline de Pré-processamento e Divisão dos Dados
- [x] Criar branch `feature/fase2-preprocessamento` a partir de `develop`.
- [x] Divisão estratificada (`stratify=y`) em Treino, Validação e Teste (ex: 70/10/20 ou 80/20).
- [x] Normalização dos pixels para o intervalo $[0.0, 1.0]$ dividindo por 255.0.
- [x] Célula conceitual: Justificar textualmente a importância da normalização para convergência de modelos lineares e distâncias métricas.
- **Branch**: `feature/fase2-preprocessamento`
- **Commit Planejado**: `implementa split estratificado e normalizacao de dados`

---

### [ ] Passo 4: Fase 3 - Implementação e Treinamento de 3 Modelos Distintos
- [ ] Criar branch `feature/fase3-modelagem` a partir de `develop`.
- [ ] Implementar Modelo 1: Random Forest Classifier (Ajuste de `n_estimators` e `max_depth`).
- [ ] Implementar Modelo 2: Support Vector Machine (SVM) (Ajuste de `C` e `kernel`).
- [ ] Implementar Modelo 3: Multi-Layer Perceptron (MLP) (Ajuste de neurônios, taxa de aprendizado e ativação).
- [ ] Registrar tempo de treinamento de cada modelo.
- **Branch**: `feature/fase3-modelagem`
- **Commit Planejado**: `implementa e treina modelos random forest, svm e mlp`

---

### [ ] Passo 5: Fase 4 - Avaliação Comparativa de Desempenho
- [ ] Criar branch `feature/fase4-avaliacao-comparativa` a partir de `develop`.
- [ ] Gerar Matrizes de Confusão completas ($10 \times 10$) em Heatmap para os 3 modelos.
- [ ] Gerar Tabela Comparativa Consolidada (Accuracy, Precision ponderada, Recall ponderado, F1-Score ponderado).
- [ ] Célula técnica de conclusão: Análise de dígitos mais confusos (ex: 4 vs 9, 3 vs 5), melhor modelo e custo computacional (tempo vs acurácia).
- **Branch**: `feature/fase4-avaliacao-comparativa`
- **Commit Planejado**: `implementa avaliacao comparativa e matrizes de confusao`

---

### [ ] Passo 6: Fase 5.1 & 5.2 - Mascaramento de Classes & Generalização Extrema (OOD)
- [ ] Criar branch `feature/fase5-generalizacao-ood` a partir de `develop`.
- [ ] Desafio A: Remover pelo menos duas classes do treino (ex: dígitos 4 e 7) e treinar modelo sem vê-las.
- [ ] Desafio B: Testar o modelo exclusivamente nas classes ocultadas.
- [ ] Plotar distribuição de predições, matriz de confusão e analisar o fenômeno de "Falsa Certeza" (Overconfidence).
- **Branch**: `feature/fase5-generalizacao-ood`
- **Commit Planejado**: `implementa testes de estresse com mascaramento e inferencia ood`

---

### [ ] Passo 7: Fase 5.3 - Desafio C: Inferência com Imagens Manuscritas Próprias
- [ ] Criar branch `feature/fase5-inferencia-customizada` a partir de `develop`.
- [ ] Criar pipeline de visão computacional (OpenCV/Pillow): Grayscale -> Inversão -> Bounding Box -> Redimensionamento $28 \times 28$ com centro de massa -> Normalização $[0, 1]$.
- [ ] Salvar amostras de imagens reais em `data/custom_digits/`.
- [ ] Realizar predição com o melhor modelo e plotar imagem processada lado a lado com gráfico de probabilidades.
- **Branch**: `feature/fase5-inferencia-customizada`
- **Commit Planejado**: `implementa pipeline de visao computacional para digitos manuscritos reais`

---

### [ ] Passo 8: Documentação Completa (README.md) & Preparação do Vídeo
- [ ] Criar branch `feature/documentacao-readme` a partir de `develop`.
- [ ] Criar `README.md` completo com escopo, arquitetura, instruções de execução, resultados e link do vídeo.
- [ ] Elaborar roteiro estruturado para o vídeo de apresentação de 10 minutos (respondendo a todos os itens da Seção 5.4).
- [ ] Realizar merges finais: todas as features em `develop`, e `develop` na `main`.
- **Branch**: `feature/documentacao-readme` -> `develop` -> `main`
- **Commit Planejado**: `adiciona documentacao completa do projeto no readme`

---

## 📜 Histórico de Commits e Rastreabilidade

| Data / Hora | Branch | Hash | Mensagem do Commit | Descrição da Entrega |
| :--- | :--- | :--- | :--- | :--- |
| 31/08/2026 | `main` | `283c030` | `configura estrutura inicial do projeto e dependencias` | Estrutura base, `.gitignore` e `requirements.txt`. |
| 31/08/2026 | `develop` | `(próximo)` | `adiciona quadro kanban de planejamento do projeto` | Criação do arquivo de rastreamento KANBAN.md. |

