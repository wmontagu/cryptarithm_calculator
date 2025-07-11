import time
import itertools

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Easy
def operate(int1, int2, int3, operation):
    if operation == "+":
        if (int1 + int2) == int3:
            return True
    if operation == "-":
        if (int1 - int2) == int3:
            return True
    if operation == "*":
        if (int1 * int2) == int3:
            return True
    if operation == "/":
        if (int1 / int2) == int3:
            return True
        
def solve_crypto_init_easy(str1, str2, equal, operation):
    d = set()
    dicti = []
    for l in str1:
        if [l, 0] not in dicti:
            dicti.append([l, 0])
    for l in str2:
        if [l, 0] not in dicti:
            dicti.append([l, 0])
    for l in equal:
        if [l, 0] not in dicti:
            dicti.append([l, 0])
    
    r = len(dicti)
    fullList = []
    keepstr1eq = str1
    keepstr2eq = str2
    keepstr3eq = equal
    combos = list(itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], r))
    for combo in combos:
        fullList += (list(itertools.permutations(combo)))
    for t in fullList:
        for i in range(len(t)):
            str1 = str1.replace(dicti[i][0], str(t[i]))
            str2 = str2.replace(dicti[i][0], str(t[i]))
            equal = equal.replace(dicti[i][0], str(t[i]))
            dicti[i][1] = t[i]
        if operate(int(str1), int(str2), int(equal), operation):
            if str1[0] != '0' and str2[0] != '0' and equal[0] != '0':
                return (str1, str2, equal)
        str1 = keepstr1eq
        str2 = keepstr2eq
        equal = keepstr3eq
    return "No solution found"
    
        
        
# Medium

def init_crypto_variables(str1, str2, equal):
    dicti = dict()
    L = []
    for l in str1:
        if l not in L:
            L.append(l)
    for l in str2:
        if l not in L:
            L.append(l)
    for l in equal:
        if l not in L:
            L.append(l)
    
    for l in L:
        dicti[l] = numbers[:]
    
    #Initialization making it so first digit cannot be 0.
    if str1 and str1[0] in dicti:
        dicti[str1[0]] = numbers[1:]
    if str2 and str2[0] in dicti:
        dicti[str2[0]] = numbers[1:]
    if equal and equal[0] in dicti:
        dicti[equal[0]] = numbers[1:]
    
    return L, dicti

class CSP:
    def __init__(self, variables, domains, str1, str2, equal, operation):
        self.variables = variables
        self.domains = {var: domain[:] for var, domain in domains.items()}
        self.assignments = {}
        self.str1 = str1
        self.str2 = str2
        self.equal = equal
        self.operation = operation

    def is_assignment_consistent(self, var, value):
        for assigned_var, assigned_value in self.assignments.items():
            if assigned_var != var and assigned_value == value:
                return False
        return True
    
    def is_complete(self):
        return len(self.assignments) == len(self.variables)
    
    def select_unassigned(self):
        unassigned = [var for var in self.variables if var not in self.assignments]
        if not unassigned:
            return None
        # MRV
        return min(unassigned, key=lambda var: len(self.domains[var]))
    
    def forward_check(self, var, value):
        removed = {}
        for other_var in self.variables:
            if other_var != var and other_var not in self.assignments:
                removed[other_var] = []
                if value in self.domains[other_var]:
                    self.domains[other_var].remove(value)
                    removed[other_var].append(value)
                if not self.domains[other_var]:
                    return False, removed
        return True, removed
    
    def restore_domains(self, removed):
        for var, values in removed.items():
            self.domains[var].extend(values)
    
    def check_arithmetic_constraint(self):
        """Check if current complete assignment satisfies the arithmetic equation"""
        def word_to_number(word):
            return int(''.join(str(self.assignments[char]) for char in word))
        
        num1 = word_to_number(self.str1)
        num2 = word_to_number(self.str2)
        result = word_to_number(self.equal)
        
        if self.operation == '+':
            return (num1 + num2) == result
        elif self.operation == '-':
            return (num1 - num2) == result
        elif self.operation == '*':
            return (num1 * num2) == result
        elif self.operation == '/':
            return num2 != 0 and (num1 // num2) == result
        return False
    
    def backtrack(self):
        if self.is_complete():
            return self.check_arithmetic_constraint()

        var = self.select_unassigned()
        if var is None:
            return True
        
        for value in self.domains[var][:]:  
            if self.is_assignment_consistent(var, value):

                self.assignments[var] = value
                

                consistent, removed = self.forward_check(var, value)
                
                if consistent:
                    result = self.backtrack()
                    if result:
                        return True
                

                self.restore_domains(removed)
                del self.assignments[var]
        
        return False

def solve_crypto_init_medium(str1, str2, equal, operation):
    variables, domains = init_crypto_variables(str1, str2, equal)
    csp = CSP(variables, domains, str1, str2, equal, operation)
    
    if csp.backtrack():
        return csp.assignments
    else:
        return None


def print_solution_medium(assignment, str1, str2, result_str, operation):
    if assignment:
        print("Solution found:")
        for var in sorted(assignment.keys()):
            print(f"{var} = {assignment[var]}")
        
        def word_to_number(word):
            return int(''.join(str(assignment[char]) for char in word))
        
        num1 = word_to_number(str1)
        num2 = word_to_number(str2)
        result = word_to_number(result_str)
        
        print(f"\n{str1:>6} = {num1}")
        print(f"{str2:>6} = {num2}")
        print("-" * 8)
        print(f"{result_str:>6} = {result}")
        print(f"\nVerification: {num1} {operation} {num2} = {result} (correct)")
    else:
        print("No solution found")


# Hard

def performance_test():
    print("=== PERFORMANCE COMPARISON ===")
    

    print("\nTesting CSP method...")
    start_time = time.time()
    csp_solution = solve_crypto_init_medium("SEND", "MORE", "MONEY", '+')
    csp_time = time.time() - start_time
    
    print(f"CSP Runtime: {csp_time:.4f} seconds")
    if csp_solution:
        print("CSP Solution found!")
        print_solution_medium(csp_solution, "SEND", "MORE", "MONEY", '+')
    
    print("\n" + "="*50)
    
    print("\nTesting Brute Force method...")
    start_time = time.time()
    easy_solution = solve_crypto_init_easy("SEND", "MORE", "MONEY", '+')
    brute_time = time.time() - start_time
    
    print(f"Brute Force Runtime: {brute_time:.4f} seconds")
    print(f"Brute Force Result: {easy_solution}")
    
    print(f"\nSpeedup: {brute_time/csp_time:.1f}x faster with CSP")




# Verification code moved to a function to prevent running on import
def run_verification_test():
    print("\n" + "="*60)
    print("TESTING SIMPLER PROBLEM: TWO + TWO = FOUR")
    print("="*60)

    start_time = time.time()
    simple_csp = solve_crypto_init_medium("TWO", "TWO", "FOUR", '+')
    simple_csp_time = time.time() - start_time
    print(f"CSP time for simpler problem: {simple_csp_time:.4f} seconds")
    print_solution_medium(simple_csp, "TWO", "TWO", "FOUR", '+')

# Uncomment to run the test manually
# if __name__ == "__main__":
#     run_verification_test()






