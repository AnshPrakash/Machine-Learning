Y: (binary) default payment (Yes = 1, No = 0), as the response variable


X1: (continuous) Amount of the given credit
X2: (binary) Gender (Can take values: 1, 2)
X3: (categorical) Education (Can take values: 0, 1, 2, 3, 4, 5, 6).
X4: (categorical) Marital status (Can take values: 0, 1, 2, 3).
X5: (continuous) Age


X6 - X11: (categorical) History of past monthly payment records (from April to September, 2005) as follows: 
	
	X6 = the repayment status in September, 2005; 
	X7 = the repayment status in August, 2005; 
	.
	.
	.
	X11 = the repayment status in April, 2005. 

The measurement scale for the repayment status is: 

	-2 = early payment
	-1 = early payment
	0 = pay duly; 
	1 = payment delay for one month; 
	2 = payment delay for two months;
	.
	.
	. 
	8 = payment delay for eight months; 
	9 = payment delay for nine months and above


X12-X17: (continuous) Amount of bill statement 
	
	X12 = amount of bill statement in September, 2005; 
	X13 = amount of bill statement in August, 2005;
	.
	.
	. 
	X17 = amount of bill statement in April, 2005.


X18-X23: (continuous) Amount of previous payment 
	
	X18 = amount paid in September, 2005; 
	X19 = amount paid in August, 2005;
	.
	.
	.
	X23 = amount paid in April, 2005. 