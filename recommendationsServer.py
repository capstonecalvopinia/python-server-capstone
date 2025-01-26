from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Obtener la ruta del directorio donde está el script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Cargar el modelo para recomendaciones generales
recommendation_model_path = os.path.join(script_dir, 'model', 'random_forest_model.pkl')
print("Cargando el modelo general...")
recommendation_model = joblib.load(recommendation_model_path)
print("Modelo general cargado exitosamente.")

# Cargar el modelo para recomendaciones basadas en el carrito
cart_model_path = os.path.join(script_dir, 'model', 'complementary_product_model.pkl')
print("Cargando el modelo de productos complementarios...")
cart_model = joblib.load(cart_model_path)
print("Modelo de productos complementarios cargado exitosamente.")

# Cargar el modelo para recomendaciones basadas en recetas
recipe_model_path = os.path.join(script_dir, 'model', 'recipe_recommender.pkl')
print("Cargando el modelo de recetas...")
recipe_model = joblib.load(recipe_model_path)
print("Modelo de recetas cargado exitosamente.")

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     """
#     Endpoint para generar recomendaciones generales basadas en el historial del usuario.
#     """
#     try:
#         data = request.json
#         user_id = data['user_id']
#         catalog_data = pd.DataFrame(data['catalog_data'])
#         history_data = pd.DataFrame(data['history_data'])

#         user_history = history_data[history_data['user_id'] == user_id]
#         purchased_products = set(user_history['product_id'])

#         catalog_features = catalog_data.copy()
#         catalog_features = pd.get_dummies(catalog_features, columns=['category'])
#         catalog_features = catalog_features.loc[~catalog_features['product_id'].isin(purchased_products)]

#         X_recommendation = pd.DataFrame({
#             'user_id': [user_id] * len(catalog_features),
#             'product_id': catalog_features['product_id']
#         })
#         X_recommendation = pd.get_dummies(X_recommendation, columns=['user_id', 'product_id'])

#         missing_cols = set(recommendation_model.feature_names_in_) - set(X_recommendation.columns)
#         for col in missing_cols:
#             X_recommendation[col] = 0
#         X_recommendation = X_recommendation[recommendation_model.feature_names_in_]

#         probabilities = recommendation_model.predict_proba(X_recommendation)[:, 1]
#         catalog_features['purchase_probability'] = probabilities

#         recommended_products = catalog_features.nlargest(5, 'purchase_probability')
#         return jsonify({
#             "status": True,
#             "msg": "Recomendaciones generadas con éxito.",
#             "recommendations": recommended_products[['product_id', 'purchase_probability']].to_dict(orient='records')
#         })
#     except Exception as e:
#         return jsonify({"status": False, "msg": str(e), "recommendations": []}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Endpoint para generar recomendaciones generales basadas en el historial del usuario.
    """
    try:
        # Obtener datos de la solicitud
        data = request.json
        user_id = data['user_id']
        catalog_data = pd.DataFrame(data['catalog_data'])
        history_data = pd.DataFrame(data['history_data'])

        # Filtrar historial del usuario
        user_history = history_data[history_data['user_id'] == user_id]
        purchased_products = set(user_history['product_id'])

        # Preparar características del catálogo
        catalog_features = catalog_data.copy()
        catalog_features = pd.get_dummies(catalog_features, columns=['category'])
        catalog_features = catalog_features.loc[~catalog_features['product_id'].isin(purchased_products)]

        # Crear DataFrame para recomendaciones
        X_recommendation = pd.DataFrame({
            'user_id': [user_id] * len(catalog_features),
            'product_id': catalog_features['product_id']
        })
        X_recommendation = pd.get_dummies(X_recommendation, columns=['user_id', 'product_id'])

        # Manejar columnas faltantes de manera eficiente
        missing_cols = list(set(recommendation_model.feature_names_in_) - set(X_recommendation.columns))
        if missing_cols:
            missing_df = pd.DataFrame(0, index=X_recommendation.index, columns=missing_cols)
            X_recommendation = pd.concat([X_recommendation, missing_df], axis=1)

        # Reordenar columnas para que coincidan con las del modelo
        X_recommendation = X_recommendation[recommendation_model.feature_names_in_]

        # Calcular probabilidades de compra
        probabilities = recommendation_model.predict_proba(X_recommendation)[:, 1]
        catalog_features['purchase_probability'] = probabilities

        # Seleccionar los 5 productos con mayor probabilidad de compra
        recommended_products = catalog_features.nlargest(5, 'purchase_probability')

        # Respuesta del API
        return jsonify({
            "status": True,
            "msg": "Recomendaciones generadas con éxito.",
            "recommendations": recommended_products[['product_id', 'purchase_probability']].to_dict(orient='records')
        })
    except Exception as e:
        print("Error: ", e)
        return jsonify({
            "status": False,
            "msg": str(e),
            "recommendations": []
        }), 500


@app.route('/cart-recommendations', methods=['POST'])
def cart_recommendations():
    """
    Endpoint para generar recomendaciones de productos complementarios basadas en un carrito.
    """
    try:
        data = request.json
        print("data: ", data)

        # Validar los datos de entrada
        if 'cart_data' not in data or 'catalog_data' not in data:
            return jsonify({"status": False, "msg": "Se requieren 'cart_data' y 'catalog_data'."}), 400

        cart_data = pd.DataFrame(data['cart_data'])
        catalog_data = pd.DataFrame(data['catalog_data'])

        if cart_data.empty or catalog_data.empty:
            return jsonify({"status": False, "msg": "Los datos del carrito o catálogo no pueden estar vacíos."}), 400

        # Validar que el catálogo contenga las columnas necesarias
        required_columns = {'product_id', 'product_price', 'product_stock', 'category'}
        if not required_columns.issubset(catalog_data.columns):
            return jsonify({
                "status": False,
                "msg": f"El catálogo debe contener las columnas: {required_columns}"
            }), 400

        # Filtrar los productos en el carrito
        cart_products = set(cart_data['product_id'])

        # Procesar el catálogo de productos
        catalog_features = pd.get_dummies(catalog_data, columns=['category'])

        # Asegurarse de que las columnas coincidan con las del modelo
        all_columns = cart_model.feature_names_in_
        catalog_features = catalog_features.reindex(columns=all_columns, fill_value=0)

        # Filtrar productos que no están en el carrito
        cart_features = catalog_features.loc[~catalog_features['product_id'].isin(cart_products)]

        # Validar que haya productos para recomendar
        if cart_features.empty:
            return jsonify({
                "status": True,
                "msg": "No hay productos para recomendar.",
                "recommendations": []
            })

        # Obtener las probabilidades de la clase positiva
        try:
            probabilities = cart_model.predict_proba(cart_features)[:, 1]
        except ValueError as e:
            print("Error al predecir probabilidades:", e)
            return jsonify({
                "status": False,
                "msg": "Error al predecir probabilidades. Verifica las características del catálogo.",
                "recommendations": []
            }), 500

        # Agregar las probabilidades al catálogo filtrado
        cart_features['complementary_probability'] = probabilities

        # Ordenar productos por relevancia y seleccionar los mejores
        recommended_products = cart_features.nlargest(5, 'complementary_probability')

        # Preparar la respuesta
        return jsonify({
            "status": True,
            "msg": "Recomendaciones generadas con éxito.",
            "recommendations": recommended_products[['product_id', 'complementary_probability']].to_dict(orient='records')
        })

    except KeyError as e:
        print("Error en los datos de entrada:", e)
        return jsonify({"status": False, "msg": f"Falta la columna: {e}", "recommendations": []}), 400

    except Exception as e:
        print("Error inesperado:", e)
        return jsonify({"status": False, "msg": "Error inesperado en el servidor.", "recommendations": []}), 500

@app.route('/recipe-recommendations', methods=['POST'])
def recipe_recommendations():
    """
    Endpoint para generar recomendaciones de recetas basadas en un producto de interés.
    """
    try:
        data = request.json
        product_id = data['product_id']
        catalog_data = pd.DataFrame(data['catalog_data'])
        recipes_data = pd.DataFrame(data['recipes_data'])
        product_recipe_data = pd.DataFrame(data['product_recipe_data'])

        # Filtrar los productos que están en el catálogo
        catalog_data = pd.get_dummies(catalog_data, columns=['category'])
        
        # Filtrar las recetas relacionadas con el producto solicitado
        product_recipes = product_recipe_data[product_recipe_data['product_id'] == product_id]
        recipe_ids = product_recipes['recipe_id'].unique()

        # Preparar el catálogo para la predicción
        catalog_features = catalog_data.copy()

        # Si hay recetas asociadas al producto, proceder con la recomendación
        if len(recipe_ids) > 0:
            # Filtrar las recetas que se relacionan con el producto de interés
            related_recipes = recipes_data[recipes_data['recipe_id'].isin(recipe_ids)]
            
            # Simulación de predicción (esto debe reemplazarse por el modelo real)
            # Este es un modelo ficticio, reemplázalo con tu lógica de recomendación
            recommended_recipes = related_recipes.sample(5)  # Tomar 5 recetas aleatorias como ejemplo

            return jsonify({
                "status": True,
                "msg": "Recomendaciones generadas exitosamente para el producto",
                "recommendations": recommended_recipes[['recipe_id', 'recipe_name']].to_dict(orient='records')
            })
        else:
            return jsonify({
                "status": False,
                "msg": "No se encontraron recetas relacionadas con el producto",
                "recommendations": []
            })

    except Exception as e:
        print("Error en la generación de recomendaciones:", e)
        return jsonify({
            "status": False,
            "msg": "Error al generar recomendaciones para el producto",
            "error": str(e),
            "recommendations": []
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
