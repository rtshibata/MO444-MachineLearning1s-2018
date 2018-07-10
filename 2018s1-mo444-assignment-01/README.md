# Tarefa 01 - MO444 
### Professor: Anderson Rocha
## Instituto de Computação - UNICAMP
-----------------
### Aluno: Renato  Shibata -- RA:082674

----------------

    Abstract. ----------------------------------------------------
    
0: `gethostbyname`
1: `socket`
2: `connect`
3: `bind`

4: `listen`
    5: `accept`
    6: `send`
    7: `recv`
    8: `htons`
    9: `htons`

Nosso RA termina em **1** e **4**, desse modo, iremos explicar as funções `socket` e `listen`.

#####  socket(int *af*, int *type*, int *protocol*)
Função para criar socket, ou seja, um ponto para comunicação e retorna o arquivo descritor para este ponto. 

