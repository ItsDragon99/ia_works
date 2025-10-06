`## Tarea 1#: Dataset y balas.
### 1. Dataset

#### Tendencias

- Se observan segmentos donde **Y permanece constante** y **X disminuye progresivamente**.
    
- Ejemplo:
    
    `690  389  0 678  389  0 ... 146  389  1 133  389  1`
    
- Esto sugiere la representación de **líneas horizontales** en distintas posiciones de Y.
    
- A lo largo de cada línea, en cierto punto, las etiquetas cambian de `0` a `1`, indicando un **límite o frontera**.
    

#### Interpretación visual

- Si se grafican, los puntos forman varias líneas horizontales (en Y = 389, 722, 714, 521, 376, 327, etc.).
    
- En cada línea:
    
    - Los valores con etiqueta `0` se agrupan en un extremo.
        
    - Los valores con etiqueta `1` se agrupan en el otro extremo.
        
- La transición de `0` a `1` marca una frontera dentro de esa línea.
    

#### Posibles usos

- Dataset para **clasificación binaria en 2D**.
    
- **Entrenamiento de modelos** (perceptrón, regresión logística, SVM).
    
- **Simulación de trayectorias o colisiones**:
    
    - `0` podría representar espacio libre.
        
    - `1` podría representar una pared u obstáculo.
        
- **Definición de regiones de interés** en un plano bidimensional.
    

---
### 2. Máquina de Estados de la Bala

#### Estados

- **Movimiento (Fly)**  
    La bala avanza en línea recta por el cuadrado.
    
- **Rebote (Bounce)**  
    Cuando toca un borde, cambia su dirección.
    
- **Colisión (Hit)**  
    Si entra en una posición con `flag = 1`, se considera que impacta en zona peligrosa.
    
- **Final (End)** _(opcional)_  
    Si la bala desaparece tras el impacto.
    

#### Transiciones

- Movimiento → Rebote: si alcanza un límite del cuadrado.
    
- Movimiento → Colisión: si entra en coordenada con `flag = 1`.
    
- Rebote → Movimiento: tras cambiar de dirección.
    
- Colisión → Movimiento: si la bala atraviesa y continúa.
    
- Colisión → Final: si se destruye tras impactar.

## Tarea 2#: Programar A* con Visualización de Nodos, Cascaron A

### Tarea 3# Probar codigo 

### Tarea 4#: Modelo ELEDA
