with open('LINKS.txt', 'r') as file:
    text = file.read()
    lines = text.splitlines()

sum = 0
link_list = []
for line in lines:
  sum = sum + 1
  link_list.append(line)
