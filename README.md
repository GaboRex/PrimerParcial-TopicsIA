
# CarID: Clasificador e Identificador de Autos 🚘

## Descripción 👀

CarID es una API diseñada para clasificar e identificar marcas de autos a partir de imágenes. Utiliza un modelo de aprendizaje automático entrenado con un conjunto de datos de 4 marcas de autos iniciales, las cuales son Porsche, Lamborghini, Mercedes y Audi.

## Propósito

El propósito principal de CarID es clasificar una imagen dada y determinar a qué marca de auto pertenece. Además, CarID puede clasificar autos por las características únicas que cada marca tiene en particular.

## Cómo funciona

CarID utiliza un modelo de aprendizaje automático que ha sido entrenado con un conjunto de datos de 4 marcas de autos. Cuando se le proporciona una imagen de un auto, la API procesa la imagen y la compara con las características aprendidas del conjunto de datos. Luego, devuelve la marca del auto que mejor se ajusta a las características de la imagen proporcionada.

## Endpoints🔚

La API de CarID tiene tres endpoints principales:

1. **GET /status**: Este endpoint devuelve información importante sobre el modelo que se utilizó, información del servicio, y cuánta memoria y CPU está usando el servicio.

2. **POST /predict_and_annotate**: Este endpoint recibe una imagen y devuelve la imagen con la predicción de a qué marca pertenece, la confianza de la predicción y el tiempo de ejecución.

3. **GET /reports**: Este endpoint guarda todas las predicciones que se hacen en el POST y luego permite descargarlas como un archivo CSV con la información más relevante de cada predicción, como la marca predicha y la confianza.

## Uso

Para utilizar la API de CarID, simplemente envía una solicitud POST con la imagen que deseas clasificar. La API procesará la imagen y te devolverá la marca del auto.

## Futuras mejoras

En el futuro, planeamos expandir el conjunto de datos para incluir más marcas de autos. Esto permitirá a CarID clasificar una gama aún mayor de autos.

## Contribuciones

Las contribuciones son bienvenidas. Si tienes alguna sugerencia o mejora, no dudes en abrir un problema o hacer un pull request.

## Autor:✒️

* **Gabriel Neme** -[GaboRex](https://github.com/GaboRex)
