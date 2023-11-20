import torch

# Define the variable x with requires_grad=True to track the computation for autograd
x = torch.tensor(4.0, requires_grad = True)

# Define the function f(x) = x^2
y = x**2

# Compute the gradient of y with respect to x
y.backward()

# Access the gradient, which is stored in the .grad attribute of the variable x
derivative_at_x = x.grad

print("The derivative of x^2 at x=4 is:", derivative_at_x.item())