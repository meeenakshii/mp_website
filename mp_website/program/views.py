from django.shortcuts import render # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from django.http import HttpResponse
from io import BytesIO
import base64
import urllib.parse  # Add this import
import re

def home_view(request):
    return render(request, 'home.html')

def simplex_method(request):
    return render(request, 'simplex_method.html')

def plot_constraints(constraints, bounds, feasible_region=None, optimal_vertex=None):
    """Plots the constraints, feasible region, and optimal solution."""
    x = np.linspace(bounds[0], bounds[1], 400)
    plt.figure(figsize=(10, 8))

    # Plot constraints as lines
    for coeff, b in constraints:
        if coeff[1] != 0:  # Plot lines with a slope
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {b}")
        else:  # Vertical line
            x_val = b / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    # Highlight feasible region
    if feasible_region is not None and len(feasible_region) > 0:
        if len(feasible_region) >= 3:
            hull = ConvexHull(feasible_region)
            polygon = Polygon(feasible_region[hull.vertices], closed=True, color='lightgreen', alpha=0.5, label='Feasible Region')
            plt.gca().add_patch(polygon)
        else:
            plt.fill(*zip(*feasible_region), color='lightgreen', alpha=0.5, label='Feasible Region')

    # Highlight corner points
    if feasible_region is not None:
        for point in feasible_region:
            plt.plot(point[0], point[1], 'bo')  # Mark corners

    # Highlight the optimal solution
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')

    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear Programming: Graphical Method")
    plt.legend()
    plt.grid()

    # Save plot to a string in base64 encoding
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    plt.close()
    return uri

def solve_linear_program(c, A, b):
    """Solve the linear programming problem and plot."""
    bounds = [0, max(b)]  # Define a reasonable range for visualization
    constraints = list(zip(A, b))

    # Solve using vertices of the feasible region
    vertices = []
    num_constraints = len(A)
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            # Find intersection of two lines
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                if all(np.dot(A, vertex) <= b) and all(vertex >= 0):  # Ensure non-negativity and feasibility
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    # Filter unique vertices
    feasible_vertices = np.unique(vertices, axis=0)

    # Ensure there are enough points for the feasible region
    if len(feasible_vertices) >= 3:
        # Evaluate the objective function at each vertex
        z_values = [np.dot(c, v) for v in feasible_vertices]
        optimal_value = max(z_values)
        optimal_vertex = feasible_vertices[np.argmax(z_values)]

        plot_uri = plot_constraints(constraints, bounds, feasible_region=feasible_vertices, optimal_vertex=optimal_vertex)

        return {"optimal_point": optimal_vertex, "optimal_value": optimal_value, "plot_uri": plot_uri}
    else:
        return {"error": "Not enough points to construct a valid feasible region."}

def gp_index(request):
    if request.method == 'POST':
        obj_func = request.POST.get('objective_function').split(',')
        c = [float(x) for x in obj_func]
        
        constraints = request.POST.getlist('constraints[]')
        A = []
        b = []
        for constraint in constraints:
            coeff, bound = constraint.split('<=')
            coeffs = coeff.split(',')
            A.append([float(x) for x in coeffs])
            b.append(float(bound))
        
        result = solve_linear_program(c, A, b)
        if "error" in result:
            return HttpResponse(result["error"])
        else:
            context = {
                'optimal_point': result["optimal_point"],
                'optimal_value': result["optimal_value"],
                'plot_uri': result["plot_uri"]
            }
            return render(request, 'gp_result.html', context)
    return render(request, 'gp_index.html')

from django.shortcuts import render
from django.http import JsonResponse
import numpy as np

def simplex(c, A, b):
    num_constraints, num_variables = A.shape
    slack_vars = np.eye(num_constraints)
    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))
    obj_row = np.hstack((-c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))
    num_total_vars = num_variables + num_constraints

    while True:
        if all(tableau[-1, :-1] >= 0):
            break
        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)
        if np.all(ratios == np.inf):
            raise ValueError("The problem is unbounded.")
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
    solution = np.zeros(num_total_vars)
    for i in range(num_constraints):
        basic_var_index = np.where(tableau[i, :-1] == 1)[0]
        if len(basic_var_index) == 1 and basic_var_index[0] < num_total_vars:
            solution[basic_var_index[0]] = tableau[i, -1]
    optimal_value = tableau[-1, -1]
    return solution[:num_variables], optimal_value

def simplex_method(request):
    if request.method == "POST":
        try:
            c = np.array([float(x) for x in request.POST.get("objective_function").split(",")])
            constraints = [constraint.split('<=') for constraint in request.POST.getlist("constraints[]")]
            A = np.array([list(map(float, constraint[0].split(','))) for constraint in constraints])
            b = np.array([float(constraint[1]) for constraint in constraints])
            solution, optimal_value = simplex(c, A, b)
            return JsonResponse({"solution": solution.tolist(), "optimal_value": optimal_value})
        except Exception as e:
            return JsonResponse({"error": str(e)})
    return render(request, "simplex_method.html")


from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from scipy.optimize import linprog
import json

def solve_transportation_problem(cost_matrix, supply, demand):
    # Convert inputs to numpy arrays
    cost_matrix = np.array(cost_matrix)
    supply = np.array(supply)
    demand = np.array(demand)

    # Number of sources and destinations
    m, n = cost_matrix.shape

    # Flatten the cost matrix for linprog
    c = cost_matrix.flatten()

    # Create the inequality constraint matrix and vector
    A_eq = []
    b_eq = []

    # Supply constraints (row-wise)
    for i in range(m):
        row_constraint = [0] * (m * n)
        for j in range(n):
            row_constraint[i * n + j] = 1
        A_eq.append(row_constraint)
        b_eq.append(supply[i])

    # Demand constraints (column-wise)
    for j in range(n):
        col_constraint = [0] * (m * n)
        for i in range(m):
            col_constraint[i * n + j] = 1
        A_eq.append(col_constraint)
        b_eq.append(demand[j])

    # Solve the problem using linprog
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    # Process the solution
    if result.success:
        solution_matrix = result.x.reshape(m, n)
        return {
            "solution": solution_matrix.tolist(),  # Convert to list for JSON serialization
            "total_cost": result.fun,
            "status": result.message,
        }
    else:
        return {
            "solution": None,
            "total_cost": None,
            "status": result.message,
        }

def transportation_problem(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            cost_matrix = data.get('cost_matrix')
            supply = data.get('supply')
            demand = data.get('demand')

            print("Received POST data:", cost_matrix, supply, demand)  # Debugging
            result = solve_transportation_problem(cost_matrix, supply, demand)
            return JsonResponse(result)
        except Exception as e:
            print("Error:", str(e))  # Debugging
            return JsonResponse({"error": str(e)})
    return render(request, 'transportation_problem.html')








