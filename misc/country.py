with open('geographic_references.txt', 'r') as file_:
	geographic_references = [line.strip().lower() for line in file_ if line.strip()]
print(geographic_references, len(geographic_references))