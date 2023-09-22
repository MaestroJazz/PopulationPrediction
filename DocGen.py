#
# Helper functions to build an html page
#
DOC = ''
def startHtml():
    global DOC

    DOC = """\n<html>
                <head>
                <style>
                table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
                }
                </style>
            </head>"""

def closeHtml():
    global DOC
    DOC += '\n</html>'
    

def startBody():
    global DOC
    DOC += '\n<body>'
    

def closeBody():
    global DOC
    DOC += '\n</body>'
    

def addTitle(title):
    global DOC
    DOC += '\n<h1>' + title + '</h1>'
    

def addSubTitle(title):
    global DOC
    DOC += '\n<h2>' + title + '</h2>'

def addSmallHeading(title):
    global DOC
    DOC += '\n<h3>' + title + '</h3>'    

def addLine():
    global DOC
    DOC += '\n<hr />'
    

def addBreak():
    global DOC
    DOC += '\n<br />'
    

def startTable():
    global DOC
    DOC += '\n<table border="1px solid black">'
    

def endTable():
    global DOC
    DOC += '\n</table>'
    

def addHeaderRow(headers):
    global DOC
    DOC += '\n<tr style="background-color:#ccc">'
    
    for header in headers:
        DOC += '\n<th style="padding: 5px;">' + header + '</th>'

    DOC += '\n</tr>'
    

def addRow(columns):
    global DOC
    DOC += '\n<tr>'
    
    index = 0
    for column in columns:
        if index > 0:
            DOC += '\n<td style="padding: 5px; text-align:right">' + column + '</td>'
        else:
            DOC += '\n<td style="padding: 5px;">' + column + '</td>'
        index += 1

    DOC += '\n</tr>' 
    

def addImage(imageUrl, altText):
    global DOC
    DOC += '<image src="' + imageUrl + '" alt="' + altText + '" />'

def saveHtml(filename):
    with open(filename, 'w') as f:
        f.write(DOC)    
    