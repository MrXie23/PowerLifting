import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("training_data.csv")
X = data[["reps", "rpe"]]
y = data["percentage_1rm"]
reg = LinearRegression().fit(X, y)

reps = 12
rpe = 9
one_rep_max = 140

def predict_percentage(reps, rpe, one_rep_max):
    percentage = reg.predict(np.array([[reps, rpe]]))[0]
    closest_weight = (percentage * one_rep_max) // 2.5 * 2.5
    return percentage, closest_weight

def update(_):
    reps = int(textbox_reps.text)
    rpe = int(textbox_rpe.text)
    one_rep_max = int(textbox_one_rep_max.text)

    percentage, closest_weight = predict_percentage(reps, rpe, one_rep_max)

    text_percentage.set_text("1RM Percentage: {:.2f}%".format(percentage*100))
    text_weight.set_text("Weights: {} kg".format(closest_weight))

    predicted_point.set_data((reps, rpe))
    predicted_point.set_3d_properties(percentage)

    fig.canvas.draw_idle()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.3)

ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y)
ax.set_xlabel('Reps')
ax.set_ylabel('RPE')
ax.set_zlabel('Percentage of 1RM')
ax.set_title('Prediction Curve')

textbox_reps = TextBox(plt.axes([0.2, 0.15, 0.1, 0.05]), 'Reps', initial=str(reps))
textbox_rpe = TextBox(plt.axes([0.4, 0.15, 0.1, 0.05]), 'RPE', initial=str(rpe))
textbox_one_rep_max = TextBox(plt.axes([0.6, 0.15, 0.1, 0.05]), '1RM', initial=str(one_rep_max))
button = Button(plt.axes([0.8, 0.15, 0.1, 0.05]), 'Update')
button.on_clicked(update)

percentage, closest_weight = predict_percentage(reps, rpe, one_rep_max)

text_percentage = plt.figtext(0.2, 0.05, "1RM Percentage: {:.2f}%".format(percentage*100))
text_weight = plt.figtext(0.2, 0.01, "Weights: {} kg".format(closest_weight))

predicted_point, = ax.plot([reps], [rpe], [percentage], 'ro', markersize=8, markeredgecolor='k', label='Predicted Point')
ax.legend()

plt.show()
