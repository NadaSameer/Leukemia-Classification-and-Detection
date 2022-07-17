import sqlite3


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


con = sqlite3.connect("Leukmia.db")
c = con.cursor()
c.execute('''CREATE TABLE Users(
        Email TEXT PRIMARY KEY,
        Password CHAR(50) NOT NULL,
        Username CHAR(50) NOT NULL
        );''')
con.commit()
c.execute('''CREATE TABLE Images(
        imageID INTEGER PRIMARY KEY AUTOINCREMENT,
        Email Text NOT NULL,
        image BLOB NOT NULL ,
        Result TEXT,
        Date TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (Email)
        REFERENCES Users (Email)
        );''')
con.commit()
# datetime('now') # while inserting the date # also try current_timestamp

# c.execute("DELETE FROM Users")
# con.commit()
# c.execute("SELECT * FROM Images")
# print(c.fetchall())
# con.commit()


def insertRegister(email, passW, username):
    try:
        sqliteConnection = sqlite3.connect('SQLite_Python.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        sqlite_insert_data_query = """ INSERT INTO Users
                                  (Email, Password, USER_NAME) VALUES (?, ?, ?)"""

        data_tuple = (email, passW, username)
        cursor.execute(sqlite_insert_data_query, data_tuple)
        sqliteConnection.commit()
        print("Data inserted successfully  into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert  data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")


def insertBLOB(image, typ):
    try:
        sqliteConnection = sqlite3.connect('SQLite_Python.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        sqlite_insert_blob_query = """ INSERT INTO Images
                                  (image,Type) VALUES (?, ?)"""

        empPhoto = convertToBinaryData(image)
        # Convert data into tuple format
        data_tuple = (empPhoto, typ)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")
