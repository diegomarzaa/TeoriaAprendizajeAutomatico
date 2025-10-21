### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ b9df5cb5-cc3c-400e-9834-875e23b659ed
using CSV

# ╔═╡ 57948b1c-2937-475a-977f-cff1f5a45ebe
using HTTP

# ╔═╡ 9420c5df-1b5c-4185-b7c1-f4c1de794b39
using DataFrames

# ╔═╡ 87613936-7ddb-452e-8389-09f5f6e2eaee
using Plots

# ╔═╡ a23524a7-b4a4-4bd7-aec8-e15800823dfe
using MLJ

# ╔═╡ 319a328e-96fa-4852-a766-1de322033450
using PlutoUI

# ╔═╡ 507dba8d-bb19-4551-9c34-333a729936bf
using LinearAlgebra: norm

# ╔═╡ 50449444-1f95-11f0-3642-89804cc4dc84
# html"""
# <link rel="stylesheet" type="text/css" href="https://belmonte.uji.es/Docencia/IR2130/Teoria/mi_estilo.css" media="screen" />
# """

# ╔═╡ e83b5f25-3fd0-4b85-8295-f1b4eafa8770
import PlotlyBase

# ╔═╡ 77c442ed-f830-4f26-901a-1ed528a67d0b
import PlotlyKaleido

# ╔═╡ 64a828bf-a426-417a-8994-7661b53e22ab
import LIBSVM

# ╔═╡ d2159870-7d0e-46a7-bda4-4e259e075906
import MLJLIBSVMInterface

# ╔═╡ f996441d-9968-4e5c-b814-25bfb0413d61
plotly();

# ╔═╡ 17a8e36a-0257-45b0-9ded-a43a85979709
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ d2d91db2-0213-4329-9d6a-524b5602e987
md"""
# Máquinas de Soporte Vectorial

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 43475479-3247-4913-b355-30e4ec0240be
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 7fd1c6ad-a229-4123-bbeb-c4b55d5ffc31
md"""
## Introducción

Las máquinas de soporte vectorial (Support Vector Machines) son un modelo muy potente para tareas de clasificación.

La idea base es encontrar un hiperplano de separación entre dos clases con margen máximo.

La potencia de esta aproximación aumenta gracias a la introducción de lo que se conoce como el truco del kernel que consiste en utilizar un espacio de representación con más dimensiones que el número de características que inicialmente tiene el conjunto de datos.
"""

# ╔═╡ 782310a4-9d89-4332-9cad-b6ab46a24a2a
md"""
## Objetivos de aprendizaje

* Resumir cuál es la estrategia del algoritmo SVM.
* Interpretar en qué consiste el truco del kernel.
* Decidir cuándo puede ser útil el algoritmo SVM para tareas de clasificación.
* Plantear una solución basada en SVM.
"""

# ╔═╡ b44ddc15-a3e7-4282-ae23-18a163c3aad2
md"""
## Referencias

1. Pattern recognition and Machine Learning. Capítulos 6 y 7.
"""

# ╔═╡ 4b30ab11-03a3-4bde-bd55-1e8f737157b6
md"""
# Objetivo de las Máquinas de Soporte Vectorial
"""

# ╔═╡ 1ba4957e-ef24-43ed-915d-e5fe1ef4b2aa
md"""
## Objetivo

El objetivo de las Máquinas de Soporte Vectorial es crear un modelo de clasificación que dada una muestra me prediga a cuál de entre dos clases pertenece.

Clasificación binaria.
"""

# ╔═╡ 39859999-911e-4a27-8760-e7f6a2d9b030
md"""
## Hiperplano de separación entre dos clases

Vamos a empezar con un conjunto de datos sencillo, en dos dimensiones, y separable mediante una recta (un hiperplano en dos dimensiones)
"""

# ╔═╡ 8064744a-796c-44a7-a2ab-d60909ecbe0d
function evalua(x, θ, θ0)
	return -(x*θ[1] + θ0) / θ[2]
end;

# ╔═╡ 40151e5b-9c13-4178-9de1-0feeb4bf31ac
function genera_datos(θ, θ0, muestras)
	xs = 10 * rand(2 * muestras)
	y = [evalua(x, θ, θ0) for x in xs]
	positivos = y[1:muestras] + rand(muestras) .+ 1
	negativos = y[muestras+1:2*muestras] - rand(muestras)
	y = cat(positivos, negativos, dims=1)
	clase = cat(repeat(["positivo"], muestras), repeat(["negativo"], muestras), dims=1)
	clase = coerce(clase, Multiclass)
	datos = DataFrame(x=xs, y=y, clase=clase)
	return datos
end;

# ╔═╡ c55facf6-29ad-4cce-9f91-e1e053c7aa70
datos = genera_datos([0.1,2.0], 5.0, 5)

# ╔═╡ 27a66f32-3a6f-4ef3-a739-01b617c73cd8
md"""
## Hiperplano de separación entre dos clases
"""

# ╔═╡ 8d94edc6-042e-4932-a7ed-57d71d6f56a2
function plot_datos(datos)
	scatter(datos[datos.clase.=="positivo", :x], datos[datos.clase.=="positivo", :y], label="Positivo", color=:blue)
	scatter!(datos[datos.clase.=="negativo", :x], datos[datos.clase.=="negativo", :y], label="Negativo", color=:red)
end;

# ╔═╡ 74da6ee7-46e9-4150-87e2-581604591ea8
md"""
Visualizamos los datos, son inventados, no pongo leyendas en los ejes:
"""

# ╔═╡ 8bb58a9c-d947-4470-9348-f8dc6c4d3e11
plot_datos(datos)

# ╔═╡ 953e4a5d-1094-49cf-86df-736a4166ab00
md"""
## Hiperplano de separación entre dos clases

¿Cuál es la mejor recta que separa las muestras dejando a un lado las positivas y al otro las negativas? En principio, tenemos muchas opciones.
"""

# ╔═╡ 131778f5-3927-4daf-a5f0-1e69e65bc773
function plot_hiperplanos(datos)
	plot_datos(datos)
	plot!([0, 10], [-2,-2.5], showlegend=false, linewidth=2)
	plot!([0, 10], [-2.5, -2.4], showlegend=false, linewidth=2)
	plot!([0, 10], [-1.75, -2.75], showlegend=false, linewidth=2)
end;

# ╔═╡ bf8856f1-8e20-4288-9db8-906e12088412
plot_hiperplanos(datos)

# ╔═╡ 85589b93-c3fc-417f-bfda-0d44c9d5d4c7
md"""
## Hiperplano de separación entre dos clases

Vamos a definir como **mejor solución** aquella línea que deja un mayor **margen** entre los dos conjuntos de datos.
"""

# ╔═╡ 6fbac7ed-919b-4696-be02-9c88a7e9a3a2
md"""
## Hiperplano de separación entre dos clases

Ya tenemos definido nuestro objetivo, encontrar el hiperplano de separación en el conjunto de datos que deja el mayor margen entre las muestras de las dos clases.

"""

# ╔═╡ 69d6b42a-a8fb-4590-9c41-1a5885ffe42d
md"""
# Construir la solución
"""

# ╔═╡ 80ab3767-3a6f-4151-8c71-95b2290e40d0
md"""
## Ecuación de la recta

Los puntos de la recta (hiperplano) que estamos buscando cumplen la siguiente ecuación:

$y(x) = \theta^T x + \theta_0 = 0$
"""

# ╔═╡ 51ae1229-3bbb-46cd-8721-70b8f93e4889
md"""
!!! danger "Atención"

Cuidado $\theta$ no es el vector director de la recta, de hecho es un vector 
perpendicular a la recta (*Demostración*); y $\theta_0$ no es el corte con el eje de ordenadas.
"""

# ╔═╡ c7531d9b-19b1-4940-af5a-7adec5033039
md"""
## Ecuación de la recta
Sí que es interesante que veamos cómo calcular la distancia de la recta al 
origen, que viene dada por la siguiente expresión (*Demostración*):

$d = \frac{\theta^T x}{\|\theta\|} = \frac{-\theta_0}{\| \theta \|}$

Dónde $x$ es cualquier punto que pertenezca a la recta.
"""

# ╔═╡ 2815c0a0-c83d-4f97-9877-7e842817d6ce
md"""
!!! info "Importante"
Si multiplicamos el vector $\theta$ y el escalar $\theta_0$ por la misma 
constante:

- La dirección del vector director no cambia.
- La distancia de la recta al origen tampoco cambia.

"""

# ╔═╡ 1aff9435-6ee9-4028-aea3-482a2f46c099
md"""
## Criterio de clasificación

El siguiente paso es establecer un criterio para decidir si una muestra 
la clasificamos como positiva o negativa.
"""

# ╔═╡ ef1817d9-3e3f-44b8-bb7c-65cce84898ee
md"""
## Criterio de clasificación
La primera idea brillante es, ya que podemos ajustar $\theta$ y $\theta_0$ con 
una constante y seguir teniendo la misma recta, lo vamos a hacer de tal modo que:

Las muestras positivas cumplan:

$y(x_{+}) = \theta^T x_+ + \theta_0 \ge 1$

Y las negativas:

$y(x_{-}) = \theta^T x_- + \theta_0 \le -1$

"""

# ╔═╡ b4825656-168a-44c6-952e-5f228ae248c6
md"""
## Criterio de clasificación
Vamos intentar condensar ambas expresiones en una única expresión, para ello 
definimos:

$t_n = +1 \: si \: x_n: positiva$
$t_n = -1 \: si \: x_n: negativa$

Ahora podemos escribir:

$t_n(y(x_n)) = t_n(\theta^T x_n + \theta_0) \ge 1 \\$
"""

# ╔═╡ 8567201c-8811-4809-b3cd-a1d6ff4ae333
md"""
## Criterio de clasificación

O lo que es lo mismo:

$\boxed{t_n(\theta^T x_n + \theta_0) - 1 \ge 0}$

Donde la igualdad se cumple justo para los puntos que están en las líneas de 
margen.
"""

# ╔═╡ 4ab666f2-078c-4414-8c77-151e38882b1f
md"""
## Anchura del margen

La anchura del margen $a$ es (*Demostración*):

$a = \frac{\theta^T}{\| \theta \|} (v_+ - v_-) = \frac{1}{\| \theta \|} 
(\theta^T v_+ - \theta^T v_-)$


$t_n(\theta^T x_n + \theta_0) - 1 = 0$

Para un punto límete en la zona positiva:

$v_+ \rightarrow 1(\theta^T v_+ + \theta_0) - 1 = 0$
$\theta^T v_+ = 1 - \theta_0$
"""

# ╔═╡ 009833dd-3e5d-424c-aec0-c0a353526cc1
md"""
## Anchura del margen

Para un punto límite en la zona negativa:

$v_- \rightarrow -1(\theta^T v_- + \theta_0) - 1 = 0$
$\theta^T v_- = -1 - \theta_0$

Finalmente:

$a = \frac{1}{\| \theta \|} [1 - \theta_0 - (-\theta_0 - 1)] = \frac{2}{\| \theta \|}$
"""

# ╔═╡ d1df1292-66d0-4a60-bbaf-e71c799e7f54
md"""
## Problema de optimización

Resumiendo, la anchura que debemos maximizar es:

$a = \frac{2}{\| \theta \|}$

Pero es más interesante minimizar su inversa al cuadrado:

$h = \frac{1}{2} \| \theta \|^2$

Con las restricciones (una por cada elemento del conjunto):

$t_n(\theta^T x_n + \theta_0) - 1 \ge 0$

Que es un problema de optimización. 
"""

# ╔═╡ 54104ab5-6417-4e32-a89d-2e85852bf243
md"""
## Problema de optimización

Para solucionarlo vamos a hacer uso de los 
multiplicadores de Lagrange (*Ver Apéndice E de libro de Bishop*) lo que 
implica definir la siguiente Lagrangiana (*Ejemplo*):

$L(\theta, \theta_0, \alpha_n) = \frac{1}{2} \| \theta \|^2 - \sum_{n=1}^N \alpha_n [t_n(\theta^T x_n + \theta_0) - 1]$

Con todos los parámetros de Lagrange $\alpha_n \ge 0$.

Fíjate en que la Lagrangiana depende de $(\theta, \theta_0, \alpha_n)$, y lo 
que conocemos son las $x_n$ y $t_n$. Es un problema de optimización difícil de 
resolver.
"""

# ╔═╡ 1d7a822d-80f8-4c19-873d-06259663ff03
md"""
## Problema de optimización

$L(\theta, \theta_0, \alpha_n) = \frac{1}{2} \| \theta \|^2 - \sum_{n=1}^N \alpha_n [t_n(\theta^T x_n + \theta_0) - 1]$
"""

# ╔═╡ 86588b2e-d8de-45f7-af7a-27f88a739c2b
md"""
!!! info "Nota"
Los signos en la Lagrangiana son opuestos porque estamos maximizando la 
primera expresión, pero minimizando la segunda.
"""

# ╔═╡ 5938fc64-d79d-4068-9967-60e695cb7d7b
md"""
## Problema de optimización

El siguiente paso es encontrar los extremos de la Lagrangiana con respecto de 
$\theta$ y $\theta_0$. Derivamos la Lagrangiana con respecto a ellos, 
igualamos a cero y despejamos:

$\frac{\partial L}{\partial \theta} = \theta - \sum_{n=1}^N \alpha_n t_n x_n = 0 
\rightarrow \boxed{\theta = \sum_{n=1}^N \alpha_n t_n x_n}$
$\frac{\partial L}{\partial \theta_0} = - \sum_{n=1}^N \alpha_n t_n = 0 
\rightarrow \boxed{\sum_{n=1}^N \alpha_n t_n = 0}$

Ya tenemos parte del problema solucionado, pero no del todo, porque no conocemos el valor de las $\alpha_n$ y, de momento, no tenemos una expresión para calcular $\theta_0$.
"""

# ╔═╡ c4abe55a-dc53-44db-87f8-3894d8657eb2
md"""
## Problema de optimización

La segunda idea brillante es sustituir la expresión para $\theta$ de la anterior 
expresión en la Lagrangiana, y hacer uso de la segunda expresión, con lo que 
la Lagrangina nos queda como (*Demostrar*):

$L(\alpha_n) = -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m t_n t_m 
x_n^T x_m + \sum_{n=1}^N \alpha_n$
"""

# ╔═╡ d37e739b-051c-4566-a96b-d5f480f8b0e3
md"""
## Problema de optimización

Si comparamos las dos expresiones:

$L(\theta, \theta_0, \alpha_n) = \frac{1}{2} \| \theta \|^2 - \sum_{n=1}^N \alpha_n [t_n(\theta^T x_n + \theta_0) - 1]$

$L(\alpha_n) = -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m t_n t_m 
x_n^T x_m + \sum_{n=1}^N \alpha_n$

La segunda tiene menos parámetros desconocidos. De hecho, la nueva expresión de
la Lagrangiana es un problema cuadrático de minimización que siempre tiene
solución numérica.
"""

# ╔═╡ b76e2333-b854-414e-a293-5a3c0c12d501
md"""
## Problema de optimización

$L(\alpha_n) = -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m t_n t_m 
x_n^T x_m + \sum_{n=1}^N \alpha_n$

Se puede demostrar que este problema de optimización añade una nueva 
restricción a las dos que ya tenemos, con lo que tenemos esta serie de 
restricciones:

$\alpha_n \ge 0$
$t_n y(x_n) - 1 \ge 0$
$\alpha_n [t_n y(x_n) - 1] = 0$

La última restricción es la clave.
"""

# ╔═╡ 6be601d9-8531-4dc3-a144-109eb3404f54
md"""
## Problema de optimización

$\alpha_n [t_n y(x_n) - 1] = 0$

Para que se cumpla esta condición tenemos dos opciones:

$\alpha_n = 0$
$t_n y(x_n) - 1 = 0$

La primera opción nos dice que el vector $x_n$ (dato de entrenamiento) no contribuye 
a la solución para obtener el valor de $\theta$.

$\theta = \sum_{n=1}^N \alpha_n t_n x_n$
"""

# ╔═╡ 6d1d0839-80c9-4342-8be1-160be3e13bd8
md"""
## Problema de optimización

La segunda opción:

$t_n y(x_n) - 1 = 0$

Nos dice que el vector $x_n$ (dato de entrenamiento) está sobre la recta del 
margen, y sí contribuye a la solución de $\theta$. 

 $x_n$ **es un vector de soporte!!!**.

Con toda esta información, ahora sí, podemos calcular el valor de $\theta_0$ 
que lo tenemos aún pendiente.
"""

# ╔═╡ 4827551b-e68d-4014-8337-611c9063c8a8
md"""
## Problema de optimización

Sustituyendo en la expresión de la condición que el vector esté sobre la 
recta del margen:

$t_n (\theta^T x_n + \theta_0) = 1$

La expresión de $\theta$ (sumatorio sólo se extiende sobre los vectores de 
soporte):

$\theta = \sum_{x_m \in S} \alpha_m t_m x_m$
"""

# ╔═╡ 072b2b88-3b59-4c07-a6db-683b167fbbd8
md"""
## Problema de optimización

Tenemos:

$t_n \left[ \left( \sum_{x_m \in S} \alpha_m t_m x_m^T x_n \right) + \theta_0 \right] = 1$

Que se puede modificar para facilitar el trabajo de los algoritmos de 
optimización (*Bishop, 2006*).
"""

# ╔═╡ fc88a32b-743a-4871-b230-01a011eb525a
md"""
## Problema de optimización

Si la anterior expresión la multiplicamos por $t_n$ y sumamos sobre todos los 
vectores de soporte, podemos despejar $\theta_0$ (*Demostración*):

$\theta_0 = \frac{1}{N_S} \left[ \sum_{x_n \in S} \left( t_n - 
\sum_{x_m \in S} \alpha_m t_m x_m^T x_n \right) \right]$

Que junto a:

$\theta = \sum_{x_m \in S} \alpha_m t_m x_m$

Resuelve el problema de clasificación.
"""

# ╔═╡ ea6d7c95-821e-4aac-89e3-d31cf49df971
md"""
## Problema de optimización

En la solución sólo están implicadas las $x_n \in S$, que hemos encontrado 
resolviendo un problema de optimización cuadrático, las que hemos llamado 
vectores de soporte.

Cuando tenemos que clasificar una nueva muestra, aplicamos:

$y(x) = \theta^T x + \theta_0$

Si $y(x) > 0$ la muestra la clasificamos como positiva.

Si $y(x) <0$ la muestra la clasificamos como negativa.

El algoritmo de clasificación es muy rápido.
"""

# ╔═╡ 6d5b3e7b-7b66-4b95-9702-cf1979d14c05
md"""
## Resumen
Lo que buscamos es la recta (hiperplano) que separa las muestras y tiene mayor 
margen.

1. Hemos partido de la ecuación de la recta.
1. Hemos definido el criterio de clasificación de muestra positiva o negativa. 
1. Hemos construido la Lagrangiana $\rightarrow$ problema de optimización.
1. En la solución que encontramos sólo participan algunos de los datos dentro de todo el conjunto.
"""

# ╔═╡ bbbe0d1c-acb9-4f69-9e38-8d5b89280c03
md"""
## Resumen

Las máquinas de soporte vectorial funcionan muy bien cuando nuestro conjunto 
de entrenamiento contiene unos cuantos miles de muestras. Si tenemos muchas 
muestras, el entrenamiento puede ser lento.
"""

# ╔═╡ 1cc522fb-bc85-412f-973c-110704feb331
md"""
## Show me the code

Después de la teoría, veamos cómo utilizar la implementación de MLJ.

Primero cargamos la estructura para crear un SVM:
"""

# ╔═╡ 643abb55-37d8-4277-a3e4-3883a85c5ee3
SVC = @load SVC pkg=LIBSVM verbosity=0

# ╔═╡ 02be1603-d7bb-44bb-a2a6-371e417c046d
md"""
Una vez cargada, la instanciamos indicando que queremos usar un kernel lineal:
"""

# ╔═╡ ca98f95a-f479-43e5-bbfd-f47123ebd27d
modelo = SVC(kernel=LIBSVM.Kernel.Linear)

# ╔═╡ 2b72cfbe-d119-46db-b5ad-8b23d9685e13
md"""
¿Qué es eso del kernel?
"""

# ╔═╡ ba0a7a09-df29-4fac-b28f-b6319c88433c
md"""
## Show me the code
"""

# ╔═╡ f3a85143-8ca8-4372-8b47-4944f17be226
md"""
Ahora definimos los datos:
"""

# ╔═╡ cbd7ba6e-2ac2-431c-a01c-f4e8017edd33
X = select(datos, [:x, :y])

# ╔═╡ 581cf399-766e-42ca-9112-78f99128440e
y = datos.clase

# ╔═╡ 61ccf1b2-5590-4910-866e-ab08a7e510fb
md"""
## Show me the code
"""

# ╔═╡ 537d3854-87c8-439e-9f66-643a44a91a3b
md"""
Finalmente, creamos la maquina que contiene el modelo y los datos; esto es propio de MLJ, para que el trabajo con cualquier modelo de ML se construya y entrene siguiendo el mismo procedimiento.
"""

# ╔═╡ 47c61751-73d4-4954-940a-dea937e1e8d6
maquina = machine(modelo, X, y)

# ╔═╡ 139b5358-8313-4613-b095-72c6361ebc16
md"""
Ahora ya podemos entrenar la máquina:
"""

# ╔═╡ 6ddba88f-ab66-469f-a5dd-68dfb97b235b
fit!(maquina)

# ╔═╡ 60dad983-fdd9-4771-8301-744a5e6ae370
md"""
## Show me the code
"""

# ╔═╡ eff50047-8b7b-4170-8ff5-f6eec94e45fd
md"""
Una vez entrenada, podemos ver cuales son los vectores de soporte que ha encontrado:
"""

# ╔═╡ 2f645347-30ad-43e3-9439-b60ea0a33136
vectores_soporte = maquina.fitresult[1].SVs.X

# ╔═╡ edffe592-e174-41a2-b697-279d4b7eda37
function plot_soporte(vectores_soporte)
	scatter!(vectores_soporte[1,:], vectores_soporte[2,:], markersize=7, label="Soporte", markeralpha=0, markerstrokealpha=1, markerstrokecolor=:black)
end;

# ╔═╡ e7198063-24cf-4801-b196-02683a5e4e95
function plot_datos_soporte(datos, vectores_soporte)
	plot_datos(datos)
	plot_soporte(vectores_soporte)
end;

# ╔═╡ 594b63ee-a145-4515-b8db-76ca02a68e94
md"""
## Show me the code

Y los mostramos en una gráfica
"""

# ╔═╡ ee2743f1-c6cb-4bec-935c-b70af7c28afe
plot_datos_soporte(datos, vectores_soporte)

# ╔═╡ 83c7eeb3-55b5-4f8d-9206-a5ecb6a9fc72
md"""
## Show me the code
"""

# ╔═╡ f3bb92ad-27e8-4b58-a6dd-b4eb67c64b05
md"""
Los coeficientes encontrados son el producto de $\alpha_m t_m$ de la ecuación:

$\theta = \sum_{x_m \in S} \alpha_m t_m x_m$
"""

# ╔═╡ edf85cdc-ff72-4560-8c29-c79184bfeadc
coefs = maquina.fitresult[1].coefs

# ╔═╡ aacb9351-7d2e-4515-9f7d-f107c461676c
md"""
Por lo tanto, $\theta$ la podemos calcular como el producto:
"""

# ╔═╡ aa531714-7f84-4766-8bf8-67e3105cc3f3
θ = vectores_soporte * coefs

# ╔═╡ 4ac4c42c-789d-493a-a509-d84e697f546f
md"""
## Show me the code
"""

# ╔═╡ 1fc57d4e-e3c8-4cb9-b296-a4e719841cac
md"""
Y $\theta_0$ la podemos calcular como:
"""

# ╔═╡ 0aeb4801-9f35-44d2-90f9-5342fbc7af8a
θ0s = sign.(coefs) - vectores_soporte' * vectores_soporte * coefs

# ╔═╡ eda3131e-5d03-45de-8530-f757af210450
θ0 = sum(θ0s) / length(θ0s)

# ╔═╡ 73c85a86-b290-4c64-a6a0-825160ef6959
md"""
## Show me the code
"""

# ╔═╡ d72b00e7-7682-4183-8b8d-49889143616d
md"""
Ya tenemos todos los ingredientes para visualizarlos:
"""

# ╔═╡ 9f54f451-63b8-4399-b219-4170c260e50d
function plot_limites(datos, vectores_soporte, θ, θ0)
	min = minimum(datos.x)
	max = maximum(datos.x)
	d = 1.0 / norm(θ) # La mitad de la anchura del margen
	plot!([min, max], [evalua(min, [θ[1],θ[2]], θ0), evalua(max, [θ[1],θ[2]], θ0)], color=:black, showlegend=false)
	plot!([min, max], [evalua(min, [θ[1],θ[2]], θ0)+d, evalua(max, [θ[1],θ[2]], θ0)+d], color=:black, linestyle=:dash, showlegend=false)
	plot!([min, max], [evalua(min, [θ[1],θ[2]], θ0)-d, evalua(max, [θ[1],θ[2]], θ0)-d], color=:black, linestyle=:dash, showlegend=false)
end;

# ╔═╡ 3916ea5b-c11e-416e-b95f-760684f0f7ad
function plot_datos_soporte_limites(datos, vectores_soporte, θ, θ0)
	plot_datos(datos)
	plot_soporte(vectores_soporte)
	plot_limites(datos, vectores_soporte, θ, θ0)
end;

# ╔═╡ e72c0f9c-7233-42da-af5c-0da3348e63fd
plot_datos_soporte_limites(datos, vectores_soporte, θ, θ0)

# ╔═╡ e9bb1b66-90b7-4d2a-bd08-eeccd5b89d58
function G(datos, vectores_soporte, θ, θ0)
	min = minimum(datos.x)
	max = maximum(datos.x)
	d = 1 / norm(θ) # La mitad de la anchura del margen
	plot!([min, max], [evalua(min, [θ[1],θ[2]], θ0), evalua(max, [θ[1],θ[2]], θ0)], color=:black, showlegend=false)
	plot!([min, max], [evalua(min, [θ[1],θ[2]], θ0)+d, evalua(max, [θ[1],θ[2]], θ0)+d], color=:black, linestyle=:dash, showlegend=false)
	plot!([min, max], [evalua(min, [θ[1],θ[2]], θ0)-d, evalua(max, [θ[1],θ[2]], θ0)-d], color=:black, linestyle=:dash, showlegend=false)
end;

# ╔═╡ 71103b9c-4d79-4f0e-845b-a84e8de1d063
function plot_datos_limites(datos, vectores_soporte)
	plot_datos(datos)
	plot_limites(datos, vectores_soporte, θ, θ0)
end;

# ╔═╡ f43e4aa9-29b0-4e94-8486-17c40fc385a7
plot_datos_limites(datos, vectores_soporte)

# ╔═╡ 9b1b2fd2-d2cb-44a5-8842-be320c17b3fb
plot_datos_limites(datos, vectores_soporte)

# ╔═╡ 4c8f525a-ca10-4854-a55a-8c620a21f9e8
md"""
# Muestras solapadas
"""

# ╔═╡ 64859989-d320-44d2-97da-4c08f75e17c2
md"""
## Muestras solapadas

Antes de pasar a ver qué es el kernel, vamos a analizar el caso en el que las 
muestras están solapadas, están en la región de la clase contraria.
"""

# ╔═╡ 4fa4e2da-bdd8-4337-bcd8-568f0c546079
function datos_solapados(datos, θ, θ0)
	copia = copy(datos)
	xs = 10 * rand(2)
	y = [evalua(x, θ, θ0) for x in xs]
	positivo = y[1] + rand() .+ 1
	negativo = y[2] - rand()
	push!(copia, [xs[1], positivo, "negativo"])
	push!(copia, [xs[2], negativo, "positivo"])
	return copia
end;

# ╔═╡ 74279119-259e-4527-a6eb-bc49d7d656c9
solapados = datos_solapados(datos, [0.1,2.0], 5.0)

# ╔═╡ 7464258c-81a6-4d74-a0b7-3fc06bcdb0a7
md"""
## Muestras solapadas
"""

# ╔═╡ bb126d64-62ba-4227-9189-9eeefaf55d15
md"""
Visualizamos los datos
"""

# ╔═╡ 18dab33a-0097-45dd-bc64-e279a2f55a9d
plot_datos(solapados)

# ╔═╡ 4949384c-5eb4-4497-82b7-d5a075613fc4
md"""
## Muestras solapadas

No vamos a entrar en el detalle de la demostración (*Bishop 2006*).

La cantidad (relacionado con la distancia) que esta vez debemos minimizar es:

$C \sum_{n=1}^N \xi_n+ \frac{1}{2} \| \theta \|^2$

1.  $\xi_n = 1$ si $x_n$ está sobre la recta.
1.  $\xi_n = 0$ si $x_n$ está sobre el margen o más allá.
1.  $\xi_n < 1$ si el punto está entre el margen y la recta, lado correcto.
1.  $\xi_n > 1$ si el punto está entre el margen y la recta, lado incorrecto.
1.  $C$ es un parámetro de regularización.
"""

# ╔═╡ b5083063-0f9e-4109-9cec-1501dc0098cb
md"""
## Muestras solapadas
"""

# ╔═╡ afc5cf1a-5bd4-4b68-8b77-0b83fc53f441
maquina_solapados = machine(SVC(kernel=LIBSVM.Kernel.Linear), select(solapados, [:x, :y]), solapados.clase)

# ╔═╡ 0fb4d386-44a8-4092-bf77-a729e857609c
fit!(maquina_solapados)

# ╔═╡ 87231bf9-92ad-4730-867e-bcd0a5e093ee
md"""
## Muestras solapadas

Mostramos el resultado:
"""

# ╔═╡ 36130a1c-6fb6-4c6f-b280-87a9c2573bda
plot_datos_soporte_limites(solapados, maquina_solapados.fitresult[1].SVs.X, θ, θ0)

# ╔═╡ 72c76679-871b-4367-8d08-4abf87c8b0d7
md"""
## Muestras solapadas

Ahora vamos a ver un caso de muestras solapadas más complicado, donde los datos no son linalmente separables.
"""

# ╔═╡ 75055e24-2f7c-41eb-a643-b4f8b23cc150
md"""
# El truco del kernel
"""

# ╔═╡ 18bfb1ef-db71-455a-ac0c-00fb6db43f59
function datos_linealmente_no_separables()
	x = coerce([-4,-3,-2,2,3,4,-1,0,1], Continuous)
	y = repeat([0], 9)
	y² = [e*e for e in x]
	clase = cat(repeat(["positivo"], 6), repeat(["negativo"], 3), dims=1)
	clase = coerce(clase, Multiclass)
	return DataFrame(x = x, y = y, clase = clase), DataFrame(x = x, y = y², clase = clase)
end;

# ╔═╡ 52934098-0e3e-46e2-bc3d-d927d5c71dae
datos_no_separables, datos_separables = datos_linealmente_no_separables();

# ╔═╡ 6aae73b0-9f59-422a-a7b2-9d2c97297229
md"""
## Problemas linealmente no separable

Observa el siguiente conjunto de datos:
"""

# ╔═╡ 79d9dfc6-9f8e-4fe9-953b-b11de1232adc
plot_datos(datos_no_separables)

# ╔═╡ 263ebdd6-8461-4207-8fdd-4bca929edbd1
md"""
Fíjate que cada muestra sólo tiene valor para una característica y la 
etiqueta de la clase.
"""

# ╔═╡ e1274616-d152-4b2e-9f71-c2cdc4f3cc1f
md"""
## Problemas linealmente no separables

Ahora, vamos a hacer un truco aumentando el número de características de cada 
muestra $(x) \rightarrow (x, x^2)$:
"""

# ╔═╡ 56c22652-b91c-4f9f-b814-5144ff3c80ec
plot_datos(datos_separables)

# ╔═╡ c28301a5-b989-4448-8c5f-96d4164c35ba
md"""
## Problemas linealmente no separables

Ahora el problema sí que es linealmente separable:
"""

# ╔═╡ bfa971c4-5352-40a6-92fd-eb0d8a0ac766
maquina_separable = machine(SVC(kernel=LIBSVM.Kernel.Linear), select(datos_separables, [:x, :y]), datos_separables.clase) |> fit!

# ╔═╡ 9d9a7d17-e611-4405-bfc5-c1d11ef99622
"""
	calcula_vectores_θ(maquina)

Calcula los vectores de soporte, parámetro θ y θ0 de la `maquina`
"""
function calcula_vectores_θ(maquina)
	vectores_soporte = maquina.fitresult[1].SVs.X
	coeficientes = maquina.fitresult[1].coefs
	θ = vectores_soporte * coeficientes
	θ0s = sign.(coeficientes) - vectores_soporte' * vectores_soporte * coeficientes
	θ0 = sum(θ0s) / length(θ0s)
	return vectores_soporte, θ, θ0
end;

# ╔═╡ 8c21e423-4d94-49d4-befe-060bebd9db13
vectores_seperable, θ_separable, θ0_separable = calcula_vectores_θ(maquina_separable);

# ╔═╡ 5ab964d3-fe4a-4caf-9b7b-f0ceb3d1e4f5
md"""
## Problemas linealmente no separables
"""

# ╔═╡ 7c46edc6-cb4d-48cb-a0fc-bfa932a19b88
plot_datos_soporte_limites(datos_separables, vectores_seperable, θ_separable, θ0_separable)

# ╔═╡ d4c709aa-7a89-4250-bd83-4b0601428d5a
md"""
Hemos ampliado el número de características en el conjunto de datos, y ahora el conjunto es separable.
"""

# ╔═╡ c729f381-b810-459c-8b16-ff88852f1e8d
md"""
## Problemas linealmente no separables

Vamos a ver un ejemplo más complejo:
"""

# ╔═╡ f1c240f7-ef5d-4296-923b-0243735112ba
function datos_circulares(muestras, separacion)
	ϕ = 2π * rand(2*muestras)
	x = cos.(ϕ)
	y = sin.(ϕ)

	x_positivas = [2 * d + sign(d) * separacion * rand() for d in x[1:muestras]]
	x_negativas = [0.5 * d + sign(d) * separacion * rand() for d in x[muestras+1:2*muestras]]
	y_positivas = [2 * d + sign(d) * separacion *rand() for d in y[1:muestras]]
	y_negativas = [0.5 * d + sign(d) * separacion * rand() for d in y[muestras+1:2*muestras]]
	x = cat(x_positivas, x_negativas, dims=1)
	y = cat(y_positivas, y_negativas, dims=1)
	clase = cat(repeat(["positivo"], muestras), repeat(["negativo"], muestras), dims=1)
	clase = coerce(clase, Multiclass)
	datos = DataFrame(x=x, y=y, clase=clase)
	
	return datos
end

# ╔═╡ 5ac9e1ca-ddd9-40ef-bbc0-0b70c7bf00ff
datos_complicados = datos_circulares(25, 0.5)

# ╔═╡ d0bfc512-2b99-4654-af36-ff0565aeacb2
md"""
## Problemas linealmente no separables
"""

# ╔═╡ 73e33858-0c2f-45a5-9063-be5417f860a0
function plot_datos_complicados(datos)
	plot_datos(datos)
	plot!(size=(600,600))
end;

# ╔═╡ 69168a15-aa7a-4951-8d47-bca2fce8b8bd
plot_datos_complicados(datos_complicados)

# ╔═╡ ac224512-fe67-44fc-bd0d-e0875a1a621e
md"""
Claramente, no es linealmente separable.
"""

# ╔═╡ 92004858-7f72-41ee-affa-645f63be89c2
md"""
## Problemas linealmente no separables

Para cada punto $p = (p_1,p_2)$ vamos a realizar la siguiente transformación:

$\phi: \mathbb{R^2} \rightarrow \mathbb{R^3}$
$(p_1, p_2) \rightarrow (p_1^2, p_2^2, \sqrt{2}p_1 p_2)$
"""

# ╔═╡ bcb95cdf-05f3-4a17-a525-92d7ab331b07
md"""
## Problemas linealmente no separables
"""

# ╔═╡ d67541ec-9f5c-45ae-98a5-a5fb46c74823
md"""
Ahora sí que podemos separar las muestras con un hiperplano:
"""

# ╔═╡ 68cbe021-7e17-48bb-875c-d938c797ecf4
function visualizacion3d(datos)
	positivos = datos[datos.clase .== "positivo", :]
	negativos = datos[datos.clase .== "negativo", :]
	x² = positivos.x .* positivos.x
	y² = positivos.y .* positivos.y
	xy = sqrt(2) .* positivos.x .* positivos.y
	scatter(x², y², xy, label="Positivos", markersize = 1, color = :blue, size=(900,600))
	x² = negativos.x .* negativos.x
	y² = negativos.y .* negativos.y
	xy = sqrt(2) .* negativos.x .* negativos.y
	scatter!(x², y², xy, label="Negativos", markersize = 1, color = :red)
end;

# ╔═╡ dc253228-bc9b-4f3b-9e32-e425525d1f2b
visualizacion3d(datos_complicados)

# ╔═╡ 51855f1c-4e7b-45a3-8de5-d3a6308f6984
md"""
## De nuevo la Lagrangiana

Por otro lado, esta es la lagrangiana que hemos minimizado:

$L(\alpha_n) = -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m t_n t_m 
x_n^T x_m + \sum_{n=1}^N \alpha_n$

El punto importante es que aparecen términos del producto escalar de dos 
puntos del conjunto de entrenamiento $x_n^T, x_m$, con:

$x_n, x_m \in \mathbb{R^D}$

Donde $D$ es la dimensión del espacio.
"""

# ╔═╡ 4678c32a-3c64-459d-bebf-d1bfd345137b
md"""
## De nuevo la Lagrangiana

Quizás, la lagrangiana no se pueda minimizar en un espacio de dimensión $D$, 
pero sí se puede minimizar en un espacio de dimensión superior:

$\phi: \mathbb{R^D} \rightarrow \mathbb{R^E}$
$x \in \mathbb{R^D} \rightarrow \phi(x) \in \mathbb{R^E}$

Con $E > D$.

$L(\alpha_n) = -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m t_n t_m 
\phi(x_n^T) \phi(x_m) + \sum_{n=1}^N \alpha_n$
"""

# ╔═╡ 0e7bf41b-3935-4f0c-a27e-a8cf17ab51f6
md"""
## El truco del kernel

El punto importe es el producto de los vectores transformados:

$\phi(x_n^T) \phi(x_m)$

Puede ocurrir que al transformar los vectores de partida lleguemos a un 
espacio con muchas más dimensiones que el original, y el producto de los 
vectores transformados sea impracticable (Ej: el kernel gaussiano transforma 
un vector de dimensión $n$ a un espacio de infinitas dimensiones).
"""

# ╔═╡ 101a42d9-b3cf-48e8-b6a9-a66cdcb25937
md"""
## El truco del kernel
El truco consiste en elegir una transformación $\phi(\cdot)$ de tal modo que 
cuando tenga que hacer el producto escalar $\phi(x_n^T) \phi(x_m)$ no tenga 
que hacer las transformaciones $\phi(\cdot)$ de cada vector porque sé como 
calcular ese producto sin tener que hacerlas.

Volvamos al ejemplo inicial:

$\phi: \mathbb{R^2} \rightarrow \mathbb{R^3}$
$x = (x_1, x_2) \rightarrow \phi(x) = (x_1^2, x_2^2, \sqrt{2}x_1 x_2)$
$x^{\prime} = (x_1^{\prime}, x_2^{\prime}) \rightarrow \phi(x^{\prime}) = 
(x_1^{\prime 2}, x_2^{\prime 2}, \sqrt{2}x_1^{\prime} x_2^{\prime})$
"""

# ╔═╡ 3c56b4fa-1953-4487-a9d7-7217296a6aa4
md"""
## El truco del kernel

Multiplicamos escalarmente los vectores transformados:

$\phi(x) \phi(x^{\prime}) = (x_1^2, x_2^2, \sqrt{2}x_1 x_2) \cdot 
(x_1^{\prime 2}, x_2^{\prime 2}, \sqrt{2}x_1^{\prime} x_2^{\prime}) =$
$x_1^2 x_1^{\prime 2} + x_2^2 x_2^{\prime 2} + 2 x_1 x_1^{\prime} x_2 x_2^{\prime} =$
$[(x_1, x_2) \cdot (x_1^{\prime}, x_2^{\prime})]^2 = k(x, x^{\prime})$

El kernel $k(x, x^{\prime})$ se calcula a partir de los vectores sin transformar, 
y el resultado es el mismo que el del producto escalar de los vectores 
transformados.
"""

# ╔═╡ e54f0557-6fb4-47b3-babf-ad26ed84bad4
md"""
## El truco del kernel

Luego:

$L(\alpha_n) = -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m t_n t_m 
\phi(x_n^T) \phi(x_m) + \sum_{n=1}^N \alpha_n$

$L(\alpha_n) = -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m t_n t_m 
k(x_n, x_m) + \sum_{n=1}^N \alpha_n$

Lo complicado es encontrar los kernels.
"""

# ╔═╡ 230bdcb8-11cb-4be8-95d2-db72d46e9b2a
md"""
## El truco del kernel

La condición (**fuerte**) que debe cumplir el kernel es que debe ser igual al 
producto escalar de los vectores transformados.

Algunos kernels:

- Lineal: $k(x,x^{\prime}) = x^t x^{\prime}$
- Polinomial: $k(x,x^{\prime}) = (\gamma x^t x^{\prime} + r)^d$
- Gaussiano: $k(x,x^{\prime}) = exp(-\gamma \|x^t x^{\prime}\|^2)$
- Sigmoide: $k(x,x^{\prime}) = tanh(\gamma x^T x^{\prime} + r)$
"""

# ╔═╡ 1d54255b-2aa3-40e7-bc54-bcd1ca83e00b
md"""
## El truco del kernel

Resulta más sencillo construir nuevos kernels a partir de kernels conocidos 
usando las siguientes reglas:

-  $k(x,x^{\prime}) = ck_1(x,x^{\prime})$
-  $k(x,x^{\prime}) = f(x)k_1(x,x^{\prime}) f(x^{\prime})$
-  $k(x,x^{\prime}) = k_1(x,x^{\prime}) + k_2(x,x^{\prime})$
-  $k(x,x^{\prime}) = k_1(x,x^{\prime}) k_2(x,x^{\prime})$
"""

# ╔═╡ 1c0b295f-ebac-4f47-8241-da3abbc17ae7
md"""
## Show me the code

Vamos a probar un kernel gaussiano:
"""

# ╔═╡ 5def91d4-b9ef-4aa6-b0eb-d2b7c4ef6b86
maquina_complicada = machine(SVC(kernel=LIBSVM.Kernel.RadialBasis), select(datos_complicados, [:x, :y]), datos_complicados.clase) |> fit!

# ╔═╡ f1ee699d-535a-436a-ba30-6face48018db
md"""
## Show me the code

Hacemos las predicciones (estamos utilizando el conjunto de datos de entrenamiento!!!), y calculamos la fracción de datos mal clasificados
"""

# ╔═╡ 53b5d2f7-2cfd-4ec3-b2c9-f9d8af30718e
ŷ = predict(maquina_complicada, select(datos_complicados, [:x, :y]))

# ╔═╡ d918ad69-de3f-4009-8527-e601368e030b
misclassification_rate(ŷ, datos_complicados.clase)

# ╔═╡ 98f63cb9-2d89-466b-8217-2ad6b982c5b9
md"""
Todos los datos se han clasificado bien. 
"""

# ╔═╡ fd82cb57-a7b4-445d-86d5-0d7aaf12f878
md"""
## Show me the code

Veamos la frontera de clasificación para este caso:
"""

# ╔═╡ c0ded5f0-d008-416d-95db-9a6b60f24f13
function clase(x,y, maquina=maquina_complicada)
	ŷ = predict(maquina, Matrix([x y]))
	if categorical(["positivo"]) == ŷ
		return 1
	else
		return 2
	end
end;

# ╔═╡ 2dceb906-68d8-4334-bc0e-254fa2f87872
function plot_kernel_gaussiano(datos, maquina)
	r = -3:0.02:3
	plot_datos(datos_complicados)
	plot_datos_soporte(datos_complicados, maquina_complicada.fitresult[1].SVs.X)
	contour!(r, r, clase, f=true, nlev=2, alpha=0.0, cbar=false, size=(600,600))
end;

# ╔═╡ 5a3c78c9-6cfa-4cde-8815-0260cf468c2e
plot_kernel_gaussiano(datos_complicados, maquina_complicada)

# ╔═╡ f92f1c74-5d7f-4c11-b140-5c8be0ae80e2
md"""
## Creatividad

¿Se te ocurre alguna otra solución ad-hoc para solucionar el problema de 
clasificación de estos datos?
"""

# ╔═╡ 0b5cd707-9937-44aa-aa9e-d02d806776e9
plot_datos_complicados(datos_complicados)

# ╔═╡ 498ef0a1-b7e1-4829-89b4-21c904a87d2d
md"""
# Aplicación
"""

# ╔═╡ 3e3792e8-da1c-48ef-a321-0dca7cb7da2e
md"""
## Aplicación al conjunto de Howell

Vamos a aplicar SVM al caso de los datos de Howell. Primero cargamos los datos:

Vamos a utilizar,
inicialmente un kernel lineal:
"""

# ╔═╡ cd24b8a6-bc56-49e2-a396-bc64fd4ad111
function carga_datos_howell()
	url = "https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Howell1.csv"
	data = CSV.File(HTTP.get(url).body) |> DataFrame
	adultos = data[data.age .> 17, [:weight, :height, :male]]
	rename!(adultos, [:x, :y, :clase])
	adultos[!, :clase] = [if x == 1 "positivo" else "negativo" end for x in adultos.clase]
	adultos[!, :clase] = coerce(adultos.clase, OrderedFactor)
	return adultos
end;

# ╔═╡ b1747147-a6e3-4060-8e2f-7df50bef2de0
adultos = carga_datos_howell()

# ╔═╡ da6e4a4d-4781-4e91-b533-2de1c5070410
md"""
## Aplicación al conjunto de Howell
"""

# ╔═╡ b142e855-46c9-47fe-a54b-ffc10eab7419
md"""
Dividimos los datos en un conjunto de entrenamiento y otro de pruebas:
"""

# ╔═╡ d07c2b6f-3145-4f2a-a9e9-acaac491621b
entrenamiento_howell, prueba_howell = partition(eachindex(adultos.clase), 0.75, shuffle=true)

# ╔═╡ 34681997-7bb6-4d89-8117-02d189b0daec
md"""
Creamos la máquina:
"""

# ╔═╡ 371e9ae2-ee34-4b76-9f3f-836b837d0e9a
maquina_howell = machine(SVC(kernel=LIBSVM.Kernel.Linear), select(adultos, [:x, :y]), adultos.clase)

# ╔═╡ a4e1a29d-8ad5-42e0-9197-021bd20df262
md"""
La entrenamos:
"""

# ╔═╡ 22f06593-18e7-40e1-a638-eb64fb3776da
fit!(maquina_howell, rows=entrenamiento_howell)

# ╔═╡ 556ee6ee-f3a7-4122-8189-626f057d4f79
md"""
## Aplicación al conjunto de Howell
"""

# ╔═╡ 28bdecaf-7954-4329-a3c5-d48c0e42e463
md"""
Estimamos la clase de los datos de prueba:
"""

# ╔═╡ b1c6dd76-0e10-4361-9549-5de0d2ad3a7e
ŷ_howell = predict(maquina_howell, rows=prueba_howell)

# ╔═╡ a3014ff8-729c-46d1-aa13-0be2490e22d5
md"""
La ratio de muestras mal clasificadas:
"""

# ╔═╡ a3976136-6172-4283-8343-2ac5d42e71d5
misclassification_rate(ŷ_howell, adultos[prueba_howell, :clase])

# ╔═╡ 1dc1e60e-d51d-493e-8cf2-a076cb114ca1
md"""
La matriz de confusión:
"""

# ╔═╡ a85d5c45-90da-4a27-9a45-e154244078db
confusion_matrix(ŷ_howell, adultos[prueba_howell, :clase])

# ╔═╡ f439edae-4570-4498-be4e-49831864f239
md"""
## Aplicación a los datos de Howell
"""

# ╔═╡ 9e783f7b-c0fd-4221-a015-45b74c76f220
md"""
Mostramos gráficamente el resultado:
"""

# ╔═╡ 777908f7-b3bc-430e-b94c-845b52cb3df1
function plot_howell_svm(adultos, maquina)
	vectores_soporte = maquina.fitresult[1].SVs.X
	coeficientes = maquina.fitresult[1].coefs
	θ = vectores_soporte * coeficientes
	# print(θ, "  ", norm(θ))
	θ0s = sign.(coeficientes) - vectores_soporte' * vectores_soporte * coeficientes
	θ0 = sum(θ0s) / length(θ0s)
	plot_datos_soporte_limites(adultos, vectores_soporte, θ, θ0)
end;

# ╔═╡ 24470b0d-35ce-45bc-9848-40a1cf8641a5
plot_howell_svm(adultos, maquina_howell)

# ╔═╡ bb88fb85-b2d4-4244-98a5-d132cee37ef7
md"""
## Aplicación a los datos de Howell
"""

# ╔═╡ e8f72d35-67a8-4672-ac17-8d12dce84f49
md"""
Ahora vamos a probar un kernel RBF (Radial Basis Function). Es muy sencillo, sólo tenemos que indicarlo en el momento de crear la máquina, y el resto del procedimiento es el mismo:
"""

# ╔═╡ fad4ea0c-39c1-4f75-91b8-2b9a877e93f6
maquina_howell_rbf = machine(SVC(kernel=LIBSVM.Kernel.RadialBasis), select(adultos, [:x, :y]), adultos.clase)

# ╔═╡ be4c0c6c-e9c6-4f38-9e5a-41d721b7bdab
md"""
Entrenamos la máquina:
"""

# ╔═╡ aa89be4d-ccd1-4086-bb16-1cad7c656b0b
fit!(maquina_howell_rbf, rows=entrenamiento_howell)

# ╔═╡ c8fa7c55-1508-434b-97f2-7216b7d683ec
md"""
## Aplicación a los datos de Howell
"""

# ╔═╡ 15c5b90e-f19a-4639-92b4-9ec4cdde09ec
md"""
Estimamos la clase de los datos de prueba:
"""

# ╔═╡ a70d2ea4-612f-4832-a55d-fe51d894631e
ŷ_howell_rbf = predict(maquina_howell_rbf, rows=prueba_howell)

# ╔═╡ 7245d4bf-9ae9-42c4-b217-45ef3ad5daf1
md"""
Mostramos la matriz de confusión, es muy parecida al caso lineal:
"""

# ╔═╡ 0185a95b-4e1e-432c-9241-3a7cd70e5d05
confusion_matrix(ŷ_howell_rbf, adultos[prueba_howell, :clase])

# ╔═╡ 7ec54078-3007-4ac3-b213-4c5bcf7b1176
md"""
La ratio de muestras mal clasificadas:
"""

# ╔═╡ ff333f42-9a30-4d38-ac92-e95e1e73f72a
misclassification_rate(ŷ_howell_rbf, adultos[prueba_howell, :clase])

# ╔═╡ ea0413eb-4c66-41ae-ac66-cdbac6129d7f
md"""
## Aplicación a los datos de Howell
"""

# ╔═╡ 08018313-909f-4050-91aa-58af6d5a7245
md"""
Y finalmente mostramos la frontera de decisión. El separador tiene aspecto de ser lineal:
"""

# ╔═╡ c89e1126-3b58-4acc-95b9-21105077300a
begin
	r1 = 30:0.1:70
	r2 = 130:0.1:180
	plot_datos(adultos)
	plot_datos_soporte(adultos, maquina_howell_rbf.fitresult[1].SVs.X)
	c(x, y) = clase(x, y, maquina_howell_rbf)
	contour!(r1, r2, c, f=true, nlev=2, alpha=0.0, cbar=false, size=(700,500))
end

# ╔═╡ 0c14eae8-a69b-4b25-a226-1cc76204c6d7
md"""
# Resumen
"""

# ╔═╡ 1fa8bfbe-a6e4-4c6e-aac0-f7041e7a56a5
md"""
## Resumen

- Las Máquinas de Soporte Vectorial fueron desarrolladas para resolver problemas de clasificación binaria.
- El objetivo es encontrar el hiperplano que separa las muestras de las dos clases, y que tiene el máximo margen entre ellas.
- El problema de separación por un hiperplano se puede extender a otras fronteras de separación introduciendo el truco del kernel.
"""

# ╔═╡ 1558e7cb-13c2-469e-a0a6-7dd4f7085b0c
md"""
## Resumen

- Las SVM funcionan muy bien cuando el número de muestras en nuestro conjunto de datos de unos cuantos miles, más allá, el algoritmo de entrenamiento puede ser muy lenta.
- La contrapartida es que la clasificación de nuevas muestras es muy rápida.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LIBSVM = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJLIBSVMInterface = "61c7150f-6c77-4bb1-949c-13197eac2a52"
PlotlyBase = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.0"
HTTP = "~1.10.16"
LIBSVM = "~0.8.1"
MLJ = "~0.20.7"
MLJLIBSVMInterface = "~0.2.1"
PlotlyBase = "~0.8.19"
PlotlyKaleido = "~2.2.6"
Plots = "~1.40.12"
PlutoUI = "~0.7.61"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "da02909b2a19e5b9c266f714f63a4b17fcb92241"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "678eb18590a8bc6674363da4d5faa4ac09c40a18"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.5.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b5bb4dc6248fde467be2a863eb8452993e74d402"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "926862f549a82d6c3a7145bc7f1adff2a91a39f0"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.15"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "0b4190661e8a4e51a842070e7dd4fae440ddb7f4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.118"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FeatureSelection]]
deps = ["MLJModelInterface", "ScientificTypesBase", "Tables"]
git-tree-sha1 = "d78c565b6296e161193eb0f053bbcb3f1a82091d"
uuid = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6"
version = "0.2.2"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "0ff136326605f8e06e9bcf085a356ab312eef18a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.13"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "9cb62849057df859575fc1dda1e91b82f8609709"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.13+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "e663925ebc3d93c1150a7570d114f9ea2f664726"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "1d4015b1eb6dc3be7e6c400fbd8042fe825a6bac"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.10"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "80d268b2f4e396edc5ea004d1e0f569231c71e9e"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.34"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LIBLINEAR]]
deps = ["Libdl", "SparseArrays", "liblinear_jll"]
git-tree-sha1 = "2cd424d3bf9b36098009df5b1f399614c12b2ee4"
uuid = "2d691ee1-e668-5016-a719-b2531b85e0f5"
version = "0.7.1"

[[deps.LIBSVM]]
deps = ["LIBLINEAR", "LinearAlgebra", "ScikitLearnBase", "SparseArrays", "libsvm_jll"]
git-tree-sha1 = "9016c6032aac779b13bbd1b3ce997606a6eb7a2b"
uuid = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
version = "0.8.1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "825289d43c753c7f1bf9bed334c253e9913997f8"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.9.0"

[[deps.LearnAPI]]
deps = ["InteractiveUtils", "Statistics"]
git-tree-sha1 = "ec695822c1faaaa64cee32d0b21505e1977b4809"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "0.1.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLFlowClient]]
deps = ["Dates", "FilePathsBase", "HTTP", "JSON", "ShowCases", "URIs", "UUIDs"]
git-tree-sha1 = "9abb12b62debc27261c008daa13627255bf79967"
uuid = "64a0f543-368b-4a9a-827a-e71edb2a0b83"
version = "0.5.1"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "FeatureSelection", "LinearAlgebra", "MLJBalancing", "MLJBase", "MLJEnsembles", "MLJFlow", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "Reexport", "ScientificTypes", "StatisticalMeasures", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "521eec7a22417d54fdc66f5dc0b7dc9628931c54"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.20.7"

[[deps.MLJBalancing]]
deps = ["MLJBase", "MLJModelInterface", "MLUtils", "OrderedCollections", "Random", "StatsBase"]
git-tree-sha1 = "f707a01a92d664479522313907c07afa5d81df19"
uuid = "45f359ea-796d-4f51-95a5-deb1a414c586"
version = "0.1.5"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "6f45e12073bc2f2e73ed0473391db38c31e879c9"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.7.0"
weakdeps = ["StatisticalMeasures"]

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatisticalMeasuresBase", "StatsBase"]
git-tree-sha1 = "84a5be55a364bb6b6dc7780bbd64317ebdd3ad1e"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.4.3"

[[deps.MLJFlow]]
deps = ["MLFlowClient", "MLJBase", "MLJModelInterface"]
git-tree-sha1 = "508bff8071d7d1902d6f1b9d1e868d58821f1cfe"
uuid = "7b7b8358-b45c-48ea-a8ef-7ca328ad328f"
version = "0.5.0"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "ad16cfd261e28204847f509d1221a581286448ae"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.6.3"

[[deps.MLJLIBSVMInterface]]
deps = ["CategoricalArrays", "LIBSVM", "MLJModelInterface", "Statistics"]
git-tree-sha1 = "4a056d2384a906ac2a2d13ed25f9092107f53eb0"
uuid = "61c7150f-6c77-4bb1-949c-13197eac2a52"
version = "0.2.1"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ceaff6618408d0e412619321ae43b33b40c1a733"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.0"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "09381923be5ed34416ed77badbc26e1adf295492"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.17.9"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase", "StatisticalMeasuresBase"]
git-tree-sha1 = "38aab60b1274ce7d6da784808e3be69e585dbbf6"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.8.8"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "4abc63cdd8dd9dd925d8e879cda280bedc8013ca"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.30"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "63603b2b367107e87dbceda4e33c67aed17e50e0"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a9697f1d06cc3eb3fb3ad49cc67f2cfabaac31ea"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "0e1340b5d98971513bddaa6bbed470670cebbbfe"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.34"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "44f6c1f38f77cafef9450ff93946c53bd9ca16ff"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyKaleido]]
deps = ["Artifacts", "Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "ba551e47d7eac212864fdfea3bd07f30202b4a5b"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.2.6"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "41c9a70abc1ff7296873adc5d768bff33a481652"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.12"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "142ee93724a9c5d04d78df7006670a93ed1b244e"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "4d083ffec53dbd5097a6723b0699b175be2b61fb"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.1.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalMeasures]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Distributions", "LearnAPI", "LinearAlgebra", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "StatisticalMeasuresBase", "Statistics", "StatsBase"]
git-tree-sha1 = "c1d4318fa41056b839dfbb3ee841f011fa6e8518"
uuid = "a19d573c-0a75-4610-95b3-7071388c7541"
version = "0.1.7"

    [deps.StatisticalMeasures.extensions]
    LossFunctionsExt = "LossFunctions"
    ScientificTypesExt = "ScientificTypes"

    [deps.StatisticalMeasures.weakdeps]
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "e4f508cf3b3253f3eb357274fe36fb3332ca9896"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "35b09e80be285516e52c9054792c884b9216ae3c"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.4.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.liblinear_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7f5f1953394b74739eaebd345f4515515a022a5b"
uuid = "275f1f90-abd2-5ca1-9ad8-abd4e3d66eb7"
version = "2.47.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libsvm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "7625dde5e9eab416c1cb791627f065ce55297eff"
uuid = "08558c22-525a-5d2a-acf6-0ac6658ffce4"
version = "3.25.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╠═50449444-1f95-11f0-3642-89804cc4dc84
# ╟─b9df5cb5-cc3c-400e-9834-875e23b659ed
# ╟─57948b1c-2937-475a-977f-cff1f5a45ebe
# ╟─9420c5df-1b5c-4185-b7c1-f4c1de794b39
# ╟─87613936-7ddb-452e-8389-09f5f6e2eaee
# ╟─e83b5f25-3fd0-4b85-8295-f1b4eafa8770
# ╟─77c442ed-f830-4f26-901a-1ed528a67d0b
# ╟─a23524a7-b4a4-4bd7-aec8-e15800823dfe
# ╟─64a828bf-a426-417a-8994-7661b53e22ab
# ╟─d2159870-7d0e-46a7-bda4-4e259e075906
# ╟─319a328e-96fa-4852-a766-1de322033450
# ╟─507dba8d-bb19-4551-9c34-333a729936bf
# ╟─f996441d-9968-4e5c-b814-25bfb0413d61
# ╟─17a8e36a-0257-45b0-9ded-a43a85979709
# ╟─d2d91db2-0213-4329-9d6a-524b5602e987
# ╟─43475479-3247-4913-b355-30e4ec0240be
# ╟─7fd1c6ad-a229-4123-bbeb-c4b55d5ffc31
# ╟─782310a4-9d89-4332-9cad-b6ab46a24a2a
# ╟─b44ddc15-a3e7-4282-ae23-18a163c3aad2
# ╟─4b30ab11-03a3-4bde-bd55-1e8f737157b6
# ╟─1ba4957e-ef24-43ed-915d-e5fe1ef4b2aa
# ╟─39859999-911e-4a27-8760-e7f6a2d9b030
# ╟─8064744a-796c-44a7-a2ab-d60909ecbe0d
# ╟─40151e5b-9c13-4178-9de1-0feeb4bf31ac
# ╟─c55facf6-29ad-4cce-9f91-e1e053c7aa70
# ╟─27a66f32-3a6f-4ef3-a739-01b617c73cd8
# ╟─8d94edc6-042e-4932-a7ed-57d71d6f56a2
# ╟─74da6ee7-46e9-4150-87e2-581604591ea8
# ╟─8bb58a9c-d947-4470-9348-f8dc6c4d3e11
# ╟─953e4a5d-1094-49cf-86df-736a4166ab00
# ╟─131778f5-3927-4daf-a5f0-1e69e65bc773
# ╟─bf8856f1-8e20-4288-9db8-906e12088412
# ╟─85589b93-c3fc-417f-bfda-0d44c9d5d4c7
# ╟─f43e4aa9-29b0-4e94-8486-17c40fc385a7
# ╟─6fbac7ed-919b-4696-be02-9c88a7e9a3a2
# ╟─69d6b42a-a8fb-4590-9c41-1a5885ffe42d
# ╟─80ab3767-3a6f-4151-8c71-95b2290e40d0
# ╟─51ae1229-3bbb-46cd-8721-70b8f93e4889
# ╟─c7531d9b-19b1-4940-af5a-7adec5033039
# ╟─2815c0a0-c83d-4f97-9877-7e842817d6ce
# ╟─1aff9435-6ee9-4028-aea3-482a2f46c099
# ╟─9b1b2fd2-d2cb-44a5-8842-be320c17b3fb
# ╟─ef1817d9-3e3f-44b8-bb7c-65cce84898ee
# ╟─b4825656-168a-44c6-952e-5f228ae248c6
# ╟─8567201c-8811-4809-b3cd-a1d6ff4ae333
# ╟─4ab666f2-078c-4414-8c77-151e38882b1f
# ╟─009833dd-3e5d-424c-aec0-c0a353526cc1
# ╟─d1df1292-66d0-4a60-bbaf-e71c799e7f54
# ╟─54104ab5-6417-4e32-a89d-2e85852bf243
# ╟─1d7a822d-80f8-4c19-873d-06259663ff03
# ╟─86588b2e-d8de-45f7-af7a-27f88a739c2b
# ╟─5938fc64-d79d-4068-9967-60e695cb7d7b
# ╟─c4abe55a-dc53-44db-87f8-3894d8657eb2
# ╟─d37e739b-051c-4566-a96b-d5f480f8b0e3
# ╟─b76e2333-b854-414e-a293-5a3c0c12d501
# ╟─6be601d9-8531-4dc3-a144-109eb3404f54
# ╟─6d1d0839-80c9-4342-8be1-160be3e13bd8
# ╟─4827551b-e68d-4014-8337-611c9063c8a8
# ╟─072b2b88-3b59-4c07-a6db-683b167fbbd8
# ╟─fc88a32b-743a-4871-b230-01a011eb525a
# ╟─ea6d7c95-821e-4aac-89e3-d31cf49df971
# ╟─6d5b3e7b-7b66-4b95-9702-cf1979d14c05
# ╟─bbbe0d1c-acb9-4f69-9e38-8d5b89280c03
# ╟─1cc522fb-bc85-412f-973c-110704feb331
# ╠═643abb55-37d8-4277-a3e4-3883a85c5ee3
# ╟─02be1603-d7bb-44bb-a2a6-371e417c046d
# ╠═ca98f95a-f479-43e5-bbfd-f47123ebd27d
# ╟─2b72cfbe-d119-46db-b5ad-8b23d9685e13
# ╟─ba0a7a09-df29-4fac-b28f-b6319c88433c
# ╟─f3a85143-8ca8-4372-8b47-4944f17be226
# ╟─cbd7ba6e-2ac2-431c-a01c-f4e8017edd33
# ╠═581cf399-766e-42ca-9112-78f99128440e
# ╟─61ccf1b2-5590-4910-866e-ab08a7e510fb
# ╟─537d3854-87c8-439e-9f66-643a44a91a3b
# ╠═47c61751-73d4-4954-940a-dea937e1e8d6
# ╟─139b5358-8313-4613-b095-72c6361ebc16
# ╠═6ddba88f-ab66-469f-a5dd-68dfb97b235b
# ╟─60dad983-fdd9-4771-8301-744a5e6ae370
# ╟─eff50047-8b7b-4170-8ff5-f6eec94e45fd
# ╠═2f645347-30ad-43e3-9439-b60ea0a33136
# ╟─edffe592-e174-41a2-b697-279d4b7eda37
# ╟─e7198063-24cf-4801-b196-02683a5e4e95
# ╟─594b63ee-a145-4515-b8db-76ca02a68e94
# ╟─ee2743f1-c6cb-4bec-935c-b70af7c28afe
# ╟─83c7eeb3-55b5-4f8d-9206-a5ecb6a9fc72
# ╟─f3bb92ad-27e8-4b58-a6dd-b4eb67c64b05
# ╠═edf85cdc-ff72-4560-8c29-c79184bfeadc
# ╟─aacb9351-7d2e-4515-9f7d-f107c461676c
# ╠═aa531714-7f84-4766-8bf8-67e3105cc3f3
# ╟─4ac4c42c-789d-493a-a509-d84e697f546f
# ╟─1fc57d4e-e3c8-4cb9-b296-a4e719841cac
# ╠═0aeb4801-9f35-44d2-90f9-5342fbc7af8a
# ╠═eda3131e-5d03-45de-8530-f757af210450
# ╟─73c85a86-b290-4c64-a6a0-825160ef6959
# ╟─3916ea5b-c11e-416e-b95f-760684f0f7ad
# ╟─d72b00e7-7682-4183-8b8d-49889143616d
# ╟─9f54f451-63b8-4399-b219-4170c260e50d
# ╟─e72c0f9c-7233-42da-af5c-0da3348e63fd
# ╟─e9bb1b66-90b7-4d2a-bd08-eeccd5b89d58
# ╟─71103b9c-4d79-4f0e-845b-a84e8de1d063
# ╟─4c8f525a-ca10-4854-a55a-8c620a21f9e8
# ╟─64859989-d320-44d2-97da-4c08f75e17c2
# ╟─4fa4e2da-bdd8-4337-bcd8-568f0c546079
# ╟─74279119-259e-4527-a6eb-bc49d7d656c9
# ╟─7464258c-81a6-4d74-a0b7-3fc06bcdb0a7
# ╟─bb126d64-62ba-4227-9189-9eeefaf55d15
# ╟─18dab33a-0097-45dd-bc64-e279a2f55a9d
# ╟─4949384c-5eb4-4497-82b7-d5a075613fc4
# ╟─b5083063-0f9e-4109-9cec-1501dc0098cb
# ╠═afc5cf1a-5bd4-4b68-8b77-0b83fc53f441
# ╠═0fb4d386-44a8-4092-bf77-a729e857609c
# ╟─87231bf9-92ad-4730-867e-bcd0a5e093ee
# ╟─36130a1c-6fb6-4c6f-b280-87a9c2573bda
# ╟─72c76679-871b-4367-8d08-4abf87c8b0d7
# ╟─75055e24-2f7c-41eb-a643-b4f8b23cc150
# ╟─18bfb1ef-db71-455a-ac0c-00fb6db43f59
# ╟─52934098-0e3e-46e2-bc3d-d927d5c71dae
# ╟─6aae73b0-9f59-422a-a7b2-9d2c97297229
# ╠═79d9dfc6-9f8e-4fe9-953b-b11de1232adc
# ╟─263ebdd6-8461-4207-8fdd-4bca929edbd1
# ╟─e1274616-d152-4b2e-9f71-c2cdc4f3cc1f
# ╟─56c22652-b91c-4f9f-b814-5144ff3c80ec
# ╟─c28301a5-b989-4448-8c5f-96d4164c35ba
# ╟─bfa971c4-5352-40a6-92fd-eb0d8a0ac766
# ╟─9d9a7d17-e611-4405-bfc5-c1d11ef99622
# ╟─8c21e423-4d94-49d4-befe-060bebd9db13
# ╟─5ab964d3-fe4a-4caf-9b7b-f0ceb3d1e4f5
# ╟─7c46edc6-cb4d-48cb-a0fc-bfa932a19b88
# ╟─d4c709aa-7a89-4250-bd83-4b0601428d5a
# ╟─c729f381-b810-459c-8b16-ff88852f1e8d
# ╟─f1c240f7-ef5d-4296-923b-0243735112ba
# ╟─5ac9e1ca-ddd9-40ef-bbc0-0b70c7bf00ff
# ╟─d0bfc512-2b99-4654-af36-ff0565aeacb2
# ╟─73e33858-0c2f-45a5-9063-be5417f860a0
# ╟─69168a15-aa7a-4951-8d47-bca2fce8b8bd
# ╟─ac224512-fe67-44fc-bd0d-e0875a1a621e
# ╟─92004858-7f72-41ee-affa-645f63be89c2
# ╟─bcb95cdf-05f3-4a17-a525-92d7ab331b07
# ╟─d67541ec-9f5c-45ae-98a5-a5fb46c74823
# ╟─68cbe021-7e17-48bb-875c-d938c797ecf4
# ╟─dc253228-bc9b-4f3b-9e32-e425525d1f2b
# ╟─51855f1c-4e7b-45a3-8de5-d3a6308f6984
# ╟─4678c32a-3c64-459d-bebf-d1bfd345137b
# ╟─0e7bf41b-3935-4f0c-a27e-a8cf17ab51f6
# ╟─101a42d9-b3cf-48e8-b6a9-a66cdcb25937
# ╟─3c56b4fa-1953-4487-a9d7-7217296a6aa4
# ╟─e54f0557-6fb4-47b3-babf-ad26ed84bad4
# ╟─230bdcb8-11cb-4be8-95d2-db72d46e9b2a
# ╟─1d54255b-2aa3-40e7-bc54-bcd1ca83e00b
# ╟─1c0b295f-ebac-4f47-8241-da3abbc17ae7
# ╠═5def91d4-b9ef-4aa6-b0eb-d2b7c4ef6b86
# ╟─f1ee699d-535a-436a-ba30-6face48018db
# ╠═53b5d2f7-2cfd-4ec3-b2c9-f9d8af30718e
# ╠═d918ad69-de3f-4009-8527-e601368e030b
# ╟─98f63cb9-2d89-466b-8217-2ad6b982c5b9
# ╟─fd82cb57-a7b4-445d-86d5-0d7aaf12f878
# ╟─c0ded5f0-d008-416d-95db-9a6b60f24f13
# ╟─2dceb906-68d8-4334-bc0e-254fa2f87872
# ╟─5a3c78c9-6cfa-4cde-8815-0260cf468c2e
# ╟─f92f1c74-5d7f-4c11-b140-5c8be0ae80e2
# ╟─0b5cd707-9937-44aa-aa9e-d02d806776e9
# ╟─498ef0a1-b7e1-4829-89b4-21c904a87d2d
# ╟─3e3792e8-da1c-48ef-a321-0dca7cb7da2e
# ╟─cd24b8a6-bc56-49e2-a396-bc64fd4ad111
# ╟─b1747147-a6e3-4060-8e2f-7df50bef2de0
# ╟─da6e4a4d-4781-4e91-b533-2de1c5070410
# ╟─b142e855-46c9-47fe-a54b-ffc10eab7419
# ╟─d07c2b6f-3145-4f2a-a9e9-acaac491621b
# ╟─34681997-7bb6-4d89-8117-02d189b0daec
# ╟─371e9ae2-ee34-4b76-9f3f-836b837d0e9a
# ╟─a4e1a29d-8ad5-42e0-9197-021bd20df262
# ╠═22f06593-18e7-40e1-a638-eb64fb3776da
# ╟─556ee6ee-f3a7-4122-8189-626f057d4f79
# ╟─28bdecaf-7954-4329-a3c5-d48c0e42e463
# ╠═b1c6dd76-0e10-4361-9549-5de0d2ad3a7e
# ╟─a3014ff8-729c-46d1-aa13-0be2490e22d5
# ╠═a3976136-6172-4283-8343-2ac5d42e71d5
# ╟─1dc1e60e-d51d-493e-8cf2-a076cb114ca1
# ╟─a85d5c45-90da-4a27-9a45-e154244078db
# ╟─f439edae-4570-4498-be4e-49831864f239
# ╟─9e783f7b-c0fd-4221-a015-45b74c76f220
# ╟─777908f7-b3bc-430e-b94c-845b52cb3df1
# ╟─24470b0d-35ce-45bc-9848-40a1cf8641a5
# ╟─bb88fb85-b2d4-4244-98a5-d132cee37ef7
# ╟─e8f72d35-67a8-4672-ac17-8d12dce84f49
# ╠═fad4ea0c-39c1-4f75-91b8-2b9a877e93f6
# ╟─be4c0c6c-e9c6-4f38-9e5a-41d721b7bdab
# ╠═aa89be4d-ccd1-4086-bb16-1cad7c656b0b
# ╟─c8fa7c55-1508-434b-97f2-7216b7d683ec
# ╟─15c5b90e-f19a-4639-92b4-9ec4cdde09ec
# ╠═a70d2ea4-612f-4832-a55d-fe51d894631e
# ╟─7245d4bf-9ae9-42c4-b217-45ef3ad5daf1
# ╠═0185a95b-4e1e-432c-9241-3a7cd70e5d05
# ╟─7ec54078-3007-4ac3-b213-4c5bcf7b1176
# ╠═ff333f42-9a30-4d38-ac92-e95e1e73f72a
# ╟─ea0413eb-4c66-41ae-ac66-cdbac6129d7f
# ╟─08018313-909f-4050-91aa-58af6d5a7245
# ╟─c89e1126-3b58-4acc-95b9-21105077300a
# ╟─0c14eae8-a69b-4b25-a226-1cc76204c6d7
# ╟─1fa8bfbe-a6e4-4c6e-aac0-f7041e7a56a5
# ╟─1558e7cb-13c2-469e-a0a6-7dd4f7085b0c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
