with open('countries.txt', 'r') as file_:
	countries = [line.strip().lower() for line in file_ if line.strip()]
print(countries, len(countries))