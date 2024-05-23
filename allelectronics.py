age = [('youth', 'no'), ('youth', 'no'), ('youth', 'yes'), ('youth', 'yes'), ('senior', 'yes'), ('senior', 'no'), ('middle_aged', 'yes'), ('youth', 'no'), ('youth', 'yes'), ('senior', 'yes'), ('youth', 'yes'), ('middle_aged', 'yes'), ('middle_aged', 'yes'), ('senior', 'no')]
income = [('high', 'no'), ('high', 'no'), ('high', 'yes'), ('medium', 'yes'), ('low', 'yes'), ('low', 'no'), ('low', 'yes'), ('medium', 'no'), ('low', 'yes'), ('medium', 'yes'), ('medium', 'yes'), ('medium', 'yes'), ('high', 'yes'), ('medium', 'no')]
student = [('no', 'no'),('no', 'no'),('no', 'yes'),('no', 'yes') ,('yes', 'yes'), ('yes', 'no'), ('yes', 'yes'), ('no', 'no'), ('yes', 'yes'),('yes', 'yes'),('yes', 'yes'),('no','yes'),('yes', 'yes'),('no','no')]
credit_rating = [('fair', 'no'), ('excellent', 'no'), ('fair', 'yes'), ('fair', 'yes'), ('fair', 'yes'), ('excellent', 'no'), ('excellent', 'yes'), ('fair', 'no'), ('fair', 'yes'), ('fair', 'yes'), ('excellent', 'yes'), ('excellent', 'yes'), ('fair', 'yes'), ('excellent', 'no')]
class1 = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

a = sum(1 for x in age if x[0] == 'youth' and x[1] == 'yes')
b = sum(1 for x in age if x[0] == 'senior' and x[1] == 'yes')
c = sum(1 for x in age if x[0] == 'middle_aged' and x[1] == 'yes')
d = sum(1 for x in income if x[0] == 'high' and x[1] == 'yes')
e = sum(1 for x in income if x[0] == 'medium' and x[1] == 'yes')
f = sum(1 for x in income if x[0] == 'low' and x[1] == 'yes')
g = sum(1 for x in student if x[0] == 'yes' and x[1] == 'yes')  
h = sum(1 for x in student if x == 'yes')
i = sum(1 for x in credit_rating if x[0] == 'fair' and x[1] == 'yes')
j = sum(1 for x in credit_rating if x[0] == 'excellent' and x[1] == 'yes')
k = sum(1 for x in class1 if x == 'no')
l = class1.count('yes')
print(f/l)