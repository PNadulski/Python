import sqlite3

class Todo:
    def __init__(self):
        self.conn = sqlite3.connect('D:/Python/SQLite3/hello.bd')
        self.c = self.conn.cursor()
        self.create_task_table()
        
    def create_task_table(self):
        self.c.execute('''CREATE TABLE IF NOT EXISTS tasks (
                     id INTEGER PRIMARY KEY,
                     name TEXT NOT NULL,
                     priority INTEGER NOT NULL
                     );''')
    
    def show_task(self):
        self.c.execute('Select * FROM tasks')
        row = self.c.fetchall()
        print(row)

    def add_task(self):
        name = input('Enter task name: ')
        priority = input('Enter priority: ')      
        self.c.execute('INSERT INTO tasks (name, priority) VALUES (?,?)', (name, priority))
        self.conn.commit()
    
    def change_priority(self):
        task_id = input('Task id: ')
        task_new_priority = input('New task priority: ')
        self.c.execute('Update tasks SET priority = ? WHERE id = ?', (task_new_priority, task_id))

    def delete_task(self):
        task_id = input('Task id: ')
        self.c.execute('DELETE FROM tasks WHERE id =?', (task_id))

app = Todo()

def menu():
    menu_choose = input('''
                            1. Show Tasks
                            2. Add Task
                            3. Change Priority
                            4. Delete Task
                            5. Exit

    ''')
    if menu_choose=='1':
        app.show_task()
        menu()
    elif menu_choose=='2':
        app.add_task()
        menu()
    elif menu_choose=='3':
        app.change_priority()
        menu()
    elif menu_choose=='4':
        app.delete_task()
        menu()
    elif menu_choose=='5':
        quit()
    else:
        print('Wrong input, pick 1-5')
menu()