from flask import Flask, render_template, request
from pydantic import BaseModel, ValidationError, StrictFloat
from typing import Literal
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

class Diamond(BaseModel):
    carat: StrictFloat
    cut: Literal['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color: Literal['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity: Literal['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    depth: StrictFloat
    table: StrictFloat
    x: StrictFloat
    y: StrictFloat
    z: StrictFloat

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = {
            'carat': float(request.form['carat']),
            'cut': request.form['cut'],
            'color': request.form['color'],
            'clarity': request.form['clarity'],
            'depth': float(request.form['depth']),
            'table': float(request.form['table']),
            'x': float(request.form['x']),
            'y': float(request.form['y']),
            'z': float(request.form['z'])
        }
        # Input values validation
        Diamond.model_validate(data)
        
        pred_pipeline = PredictionPipeline()
        prediction = pred_pipeline.initiate_prediction_pipeline(input_data=data)
        return render_template('predict.html',input_values=data, prediction=prediction)
    
    except ValidationError as e:
        return f"Validation error: {e}", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
