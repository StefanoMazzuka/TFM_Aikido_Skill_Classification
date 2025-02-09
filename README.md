# 📌 Análisis y Clasificación de Movimientos en Aikido con Aprendizaje Automático  

📅 **Año:** 2025  
👨‍💻 **Autor:** Stefano Mazzuka Cassani  
🏢 **Universidad:** Universidad Nacional de Educación a Distancia (UNED)  

Este repositorio contiene el código fuente para el análisis y clasificación de movimientos en Aikido, basado en los datos inerciales proporcionados por el trabajo 🔗 [Exploring raw data transformations on inertial sensor data to model user expertise when learning psychomotor skills](https://link.springer.com/article/10.1007/s11257-024-09393-2).
Los datos proporcionados en este repositorio son un sample muy reducido de los originales, a fin de poder ejecutar este código.

Se emplean técnicas de **Machine Learning** y **Deep Learning**, incluyendo `Gaussian Process Regression (GPR)`, `Temporal Convolutional Networks (TCN)` y `Gradient Boosted Decision Trees (GBDT)`.

---

## 🚀 **Tecnologías Utilizadas**  
Este proyecto está desarrollado en **Python 3.12.3** y usa **pip 23.2.1** para la gestión de dependencias.

📦 **Librerías principales**:
- `torch`, `torch.nn`, `torch.utils.data`, `torchmetrics`
- `pandas`, `polars`, `numpy`
- `seaborn`, `matplotlib`
- `sklearn`: `Lasso`, `GaussianProcessRegressor`, `train_test_split`, `StandardScaler`
- `imblearn`: `SMOTE`
- `tsfel`

---

## 📚 **Estructura del Proyecto**  
```
📾 Proyecto
├── 📝 app.py                  # Código principal del proyecto
├── 📝 bokken_data.csv         # Datos del ejercicio Bokken
├── 📝 shikko_data.csv         # Datos del ejercicio Shikko
├── 📝 requirements.txt        # Dependencias necesarias
├── 📝 README.md               # Documentación del proyecto
└── 📝 LICENSE                 # Licencia del proyecto
```

---

## 🛠 **Instalación y Configuración**  
### 1️⃣ Clonar el repositorio  
```bash
git clone https://github.com/StefanoMazzuka/TFM_Aikido_Skill_Classification
cd TFM_Aikido_Skill_Classification
```

### 2️⃣ Crear y activar un entorno virtual (recomendado)  
```bash
python -m venv venv
source venv/bin/activate   # En Linux/macOS
venv\Scripts\activate      # En Windows
```

### 3️⃣ Instalar dependencias  
```bash
pip install --upgrade pip==23.2.1
pip install -r requirements.txt
```

### 4️⃣ Verificar instalación  
```bash
python --version  # Debería mostrar Python 3.12.3
pip --version     # Debería mostrar pip 23.2.1
pip list          # Para ver todas las dependencias instaladas
```

---

## ▶️ **Ejecutar el Código**  
Para correr el análisis en los datasets de Aikido:  
```bash
python app.py
```
Esto ejecutará el procesamiento de datos y aplicará modelos de Machine Learning sobre los datasets `bokken_data` y `shikko_data`.

---

## 📊 **Descripción de los Modelos Implementados**  
1️⃣ **Gaussian Process Regression (GPR)**  
   - Modelo basado en procesos gaussianos para regresión de los niveles de experiencia en Aikido.
   - Se ajusta mediante el kernel RBF + optimización de hiperparámetros.

2️⃣ **Temporal Convolutional Networks (TCN)**  
   - Red neuronal convolucional para series temporales.
   - Se usa para predecir etiquetas de clasificación basadas en datos inerciales.

3️⃣ **Gradient Boosted Decision Trees (GBDT) - HyperTree**  
   - Un modelo basado en árboles de decisión para aprendizaje supervisado.
   - Aprendizaje basado en features seleccionadas mediante `LASSO`.

---

## 📚 **Licencia**

Este proyecto está licenciado bajo **MIT License**.  

🔗 [Ver licencia completa](https://opensource.org/licenses/MIT)

---

## ✅ **Resumen**  
🔹 **Python 3.12.3 y pip 23.2.1** asegurados.  
🔹 **Instalación limpia y rápida con `requirements.txt`**.  
🔹 **Modelos avanzados de ML aplicados a Aikido**.  
🔹 **Fácil ejecución con `python app.py`**.  

---

🚀 **¡Listo para usar y probar!** Si tienes preguntas, abre un *issue* en GitHub. 😊  

