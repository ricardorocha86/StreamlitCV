import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pycaret.classification import load_model, predict_model
  
st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.header('**Streamlit CV - Cúrriculo Interativo**') 
 

opcoes = ['Boas-vindas', 
		  'Produção Científica', 
		  'Playgrounds', 
		  'Outras Atividades']

pagina = st.sidebar.selectbox('Navegue pelo menu:', opcoes)
 



###### PAGINA INICIAL ######

if pagina == 'Boas-vindas':
	st.markdown('![alt text](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/Imagem8.png)')
 

	 
	st.write("""
	# Prof. Dr. Ricardo Rocha

	Olá! Muito prazer. Meu nome é Ricardo, atualmente sou docente do magistério superior na Universidade Federal da Bahia.  

	Atuo na área de Estatística Computacional junto ao [Departamento de Estatística](https://est.ufba.br/) do 
	Instituto de Matemática e Estatística da UFBA.  

	Além disso, sou o atual coordenador do [Laboratório de Estatística e Data Science (IME-UFBA)](http://led.ufba.br/), 
	onde trabalhamos em projetos com alunos, promovemos eventos e marcamos presença online. 

	O propósito dessa página é funcionar como um Currículo Iterativo. 
	Através de **Streamlit + Markdown**, podemos fazer páginas muito rapidamente, além de ter toda versatilidade do Python rodando a aplicação no back-end.

	Pretendo incluir diversas features nessa página, das quais:

	### Funcionalidades no momento
	- :heavy_check_mark:  Página de boas-vindas 
	- :heavy_check_mark:  Página de Produção Científica
	- :black_square_button:  Página de Resumo de Aulas
	- :black_square_button: Página de Atividade de Extensão
	- :heavy_check_mark:  Página de Projetos Interativos 


	Fique a vontade para entrar em contato, você pode utilizar qualquer uma das redes sociais abaixo!
 


	""")
		 
	col1, col2, col3, col4, col5 = st.beta_columns(5)
 
	col1.markdown('[![alt text](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/linkedin.png)](https://www.linkedin.com/in/ricardorocha86/)')  
	col2.markdown('[![alt text](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/github.png)](https://github.com/ricardorocha86/)')
	col3.markdown('[![alt text](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/instagram.png)](https://www.instagram.com/ricardorocha23/)')
	col4.markdown('[![alt text](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/twitter.png)](https://twitter.com/ricardorocha_86)')
	col5.markdown('[![alt text](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/gmail.png)](mailto:ricardo8610@gmail.com)')


###
elif pagina == 'Produção Científica':
	st.write("""

	# Produção Científica

	Listo aqui os 5 artigos mais relevantes da minha produção científica. 
	Agradeço a todos os parceiros de pesquisa que estiveram comigo no desenvolvimento dessas pesquisas. 
	Cito aqui apenas os artigos aceitos e publicados em periódicos internacionais.

	:star: Jeremias Leão, Marcelo Bourguignon, Diego I. Gallardo, Ricardo Rocha, Vera Tomazella. 
	**A new cure rate model with flexible competing causes with applications to melanoma and transplantation data.**  
	_Statistics in Medicine_, 2020.  
	[https://doi.org/10.1002/sim.8664](https://doi.org/10.1002/sim.8664).

	:star: Vinicius F. Calsavara, Agatha S. Rodrigues, Ricardo Rocha, Francisco Louzada, Vera Tomazella, Ana C. R. L. A. Souza. 
	**Zero-adjusted defective regression models for modeling lifetime data.**  
	_Journal of Applied Statistics_, 2019.   
	[https://doi.org/10.1080/02664763.2019.1597029](https://doi.org/10.1080/02664763.2019.1597029).

	:star: Vinicius F. Calsavara, Agatha S. Rodrigues, Ricardo Rocha, Vera Tomazella, Francisco Louzada. 
	**Defective regression models for cure rate modeling with interval‐censored data.**  
	_Biometrical Journal_, 2019.  
	[https://doi.org/10.1002/bimj.201800056](https://doi.org/10.1002/bimj.201800056).

	:star: Ricardo Rocha, Saralees Nadarajah, Vera Tomazella, Francisco Louzada. 
	**A new class of defective models based on the Marshall–Olkin family of distributions for cure rate modeling.**  
	_Computational Statistics & Data Analysis_, 2017.  
	[https://doi.org/10.1016/j.csda.2016.10.001](https://doi.org/10.1016/j.csda.2016.10.001).

	:star: Ricardo Rocha, Saralees Nadarajah, Vera Tomazella, Francisco Louzada, Amanda Eudes. 
	**New defective models based on the Kumaraswamy family of distributions with application to cancer data sets.**  
	_Statistics Methods in Medical Research_, 2015.  
	[https://doi.org/10.1177/0962280215587976](https://doi.org/10.1177/0962280215587976).




	Para mais informações, por favor, acesse meu currículo lattes para ver os demais itens de minha produção científica.
 
	[![Foo](https://raw.githubusercontent.com/ricardorocha86/StreamlitCV/main/lattes.png)](http://lattes.cnpq.br/0676420269735630)
 
    """) 
	st.markdown('---')





###### PLAYGROUND
elif pagina == 'Playgrounds':

	brinquedos = ['Regressão Linear', 
		      'Medical Cost Deploy Center']

	brinquedo = st.sidebar.selectbox('Escolha com o que quer brincar:', brinquedos)


	if brinquedo == 'Regressão Linear':
		st.markdown('![alt text](https://thumbs.dreamstime.com/b/outdoor-banner-kids-playground-equipment-bench-outdoor-banner-kids-playground-equipment-bench-flat-style-vector-116821021.jpg)')

		st.write("""
			# Brincando com uma Regressão Linear
			## Nesse brinquedo, fica fácil de visualizar as limitações do modelo de regressão linear. 
			Para isso, exploramos o caso em que há duas variáveis preditoras no problema, uma contínua e outra binária. 
			Brinque com os valores dos parâmetros que geram os dados a partir de retas.
			Veja que o modelo de regressão só apresenta bons resultados quando as retas respectivas para cada classe da variável binária são paralelas. 

		""")
		
		st.markdown('---')
		st.markdown('### Tamanho amostral e dispersão dos dados')
		
		col1, col2  = st.beta_columns(2)
		n = col1.number_input('Tamanho da amostra em cada classe', 50, 1000, 100, 50)		
		s = col2.number_input('Fator de Dispersão', 0., 5., 0.2, 0.1)		

		st.markdown('### Parâmetros das retas')
		col1, col2  = st.beta_columns(2)
		a1 = col1.number_input('Entre com o intercepto da primeira reta', -5., 5., 1., 0.2)
		a2 = col1.number_input('Entre com o intercepto da segunda reta',  -5., 5., 2., 0.2)

		b1 = col2.number_input('Entre com o coeficiente angular da primeira reta',  -5., 5., 1., 0.2)
		b2 = col2.number_input('Entre com o coeficiente angular da segunda reta',  -5., 5., 1., 0.2)
	
 
		x1 = np.random.uniform(0, 5, n)
		x2 = np.random.uniform(0, 5, n)
		y1 = a1 + b1*x1 + np.random.normal(size = n, scale = s)
		y2 = a2 + b2*x2 + np.random.normal(size = n, scale = s)

		z = {'Var1': x1, 'Var2': y1, 'Var3': np.zeros(n)}
		zeros = pd.DataFrame(z)

		u = {'Var1': x2, 'Var2': y2, 'Var3': np.ones(n)}
		uns = pd.DataFrame(u)

		df = pd.concat([zeros, uns]).round(2)
		df.reset_index(inplace = True, drop = True)
		df['Var3'] = df['Var3'].apply(int)
		print(df.sample(10))
   
		X = df[['Var1', 'Var3']]
		y = df['Var2']

		modelo = LinearRegression()
		modelo.fit(X, y)

		grid = np.arange(0, 5, 0.1)
		X1 = pd.DataFrame({'Var1': grid, 'Var3': np.zeros(len(grid))})
		X2 = pd.DataFrame({'Var1': grid, 'Var3': np.ones(len(grid))})
		y1_pred = modelo.predict(X1)
		y2_pred = modelo.predict(X2)

		plt.figure(figsize = (8, 6))
		plt.scatter(x = df['Var1'], y = df['Var2'], c = df['Var3'], cmap = 'coolwarm')
		plt.plot(grid, y1_pred, '-', color = 'blue')
		plt.plot(grid, y2_pred, '-', color = 'red')
		plt.show() 
 
		st.markdown('### Retas da regressão ajustada')

		st.pyplot()


###############################
##############################
#############################
	elif brinquedo == 'Medical Cost Deploy Center':

		modelo1 = load_model('meu-modelo-para-os-custos')
		modelo2 = load_model('meu-modelo-para-smoker')

		def classificador(modelo, dados):
			pred = predict_model(estimator = modelo, data = dados) 
			return pred

		def smap(x):  
				y = 'male' if x == 'Masculino' else 'female' 
				return y

		def rmap(x):
			if x == 'Sudeste':
				return 'southeast'
			elif x == 'Noroeste':
				return 'northwest'
			elif x == 'Sudoeste':
				return 'southwest' 
			else:
				return 'northeast'

		def fmap(x):  
			y = 'yes' if x == 'Sim' else 'no' 
			return y

		st.sidebar.header('**Medical Cost Deploy Center**') 

		opcoes = ['Página Inicial', 
				  'Modelagem de valor do seguro', 
				  'Detectar probabilidade de fraude', 
				  'Observações']

		pagina = st.sidebar.selectbox('Navegue pelo menu:', opcoes)
 



		###### PAGINA INICIAL ######

		if pagina == 'Página Inicial':
			st.markdown('![alt text](https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide1.JPG?raw=true)')
		 
			st.write("""
			# Bem-vindo ao Medical Cost Deploy Center

			Nesse Web App podemos utilizar em produção os modelos desenvolvidos tanto para
			precificar novos seguros, quanto para buscar por fraudadores do seguro.

			A lista abaixo ilustra o que está implementado até o momento. 

			### Funcionalidades no momento
			- [x]  Página Inicial 
			- [x]  Modelo em produção para precificar planos de saúde em novos clientes
			- [x]  Modelo em produção para detectar possíveis fraudadores 
			- [ ]  Deploy em lote (vários pessoas ao mesmo tempo)
			- [x]  Página de créditos 

			Os modelos desse web-app foram desenvolvidos utilizando o conjunto de 
			dados que pode ser encontrado nesse [link do kaggle](https://www.kaggle.com/mirichoi0218/insurance).

			O referencial sobre os modelos utilizados você pode encontrar nesse [link](https://github.com/gitflai/Workshop-DDS/blob/main/Dados_de_Custos_Medicos.ipynb). 
			Os modelos são desenvolvidos e analisados utilizando a biblioteca [PyCaret](https://pycaret.org/).

			Caso encontre algum erro/bug, por favor, não hesite em entrar em contato! :poop:

			Para mais informações sobre o Streamlit, consulte o [site oficial](https://www.streamlit.io/) ou a sua [documentação](https://docs.streamlit.io/_/downloads/en/latest/pdf/).

			[Lista de emojis para markdown](https://gist.github.com/rxaviers/7360908).
		 
			""")
				




		###### PAGINA: MODELO DE COTACAO DO SEGURO ######

		elif pagina == 'Modelagem de valor do seguro':
			st.markdown('![alt text](https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide2.JPG?raw=true)')

			st.markdown('# Modelagem de valor do seguro')

			st.markdown('Nessa seção é feito o deploy do modelo para cotar o valor do seguro para um indivíduo.\
					Entre com os dados e clique em APLICAR O MODELO para obter as predições.')

			st.markdown('---')

			idade = st.number_input('Idade', 18, 65, 30)
			sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])
			imc = st.number_input('Índice de Massa Corporal', 15, 54, 24)
			criancas = st.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5])
			fumante = st.selectbox("É fumante?", ['Sim', 'Não'])
			regiao = st.selectbox("Região em que mora", 
										  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])

			#custos = st.number_input('Custos da pessoa', 1000, 64000, 10000)

			dados_dicio = {'age': [idade], 'sex': [smap(sexo)], 'bmi': [imc], 
					'children': [criancas], 'region': [rmap(regiao)], 'smoker': [fmap(fumante)]}
				
			dados = pd.DataFrame(dados_dicio)

			st.markdown('---')

			if st.button('APLICAR O MODELO'):
				saida = classificador(modelo1, dados)
				pred = float(saida['Label'].round(2))
				valor = round(1.8*pred, 2)  

				s1 = 'Custo Estimado do Seguro: ${:.2f}'.format(pred)
				s2 = 'Valor de Venda do Seguro: ${:.2f}'.format(valor)

				st.markdown('## Resultados do modelo para as entradas:')
				st.write(dados)
				st.markdown('## **' + s1 + '**') 
				st.markdown('## **' + s2 + '**')  







		###### PAGINA: MODELO DE FRAUDE ######


		elif pagina == 'Detectar probabilidade de fraude':
			st.markdown('![alt text](https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide3.JPG?raw=true)')

			st.markdown('# Detectar probabilidade de fraude')

			st.markdown('Nessa seção é feito o deploy do modelo para detectar probabilidade de fraude na \
				     variável "fumante". Entre com os dados do indivíduo\
				      em análise e clique em APLICAR O MODELO para obter as predições.')

			st.markdown('---')

			idade = st.number_input('Idade', 18, 65, 30)
			sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])
			imc = st.number_input('Índice de Massa Corporal', 15, 54, 24)
			criancas = st.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5])
			#fumante = st.selectbox("É fumante?", ['Sim', 'Não'])
			regiao = st.selectbox("Região em que mora", 
										  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])

			custos = st.number_input('Custos da pessoa', 1000, 64000, 10000)
		 
			dados_dicio = {'age': [idade], 'sex': [smap(sexo)], 'bmi': [imc], 
					'children': [criancas], 'region': [rmap(regiao)], 'charges': [custos]}
				
			dados = pd.DataFrame(dados_dicio)

			st.markdown('---')

			if st.button('APLICAR O MODELO'):
				saida = classificador(modelo2, dados)
				resp = 'NÃO' if saida['Label'][0] == 'no' else 'SIM' 
				prob = saida['Score'][0] 
				st.markdown('## **O indivíduo em análise é fumante?**')
				s = 'Resposta do modelo: {}, com probabilidade {:.2f}%.'.format(resp, 100*prob)
				st.markdown('## **' + s + '**') 

				if resp == 'NÃO':
					st.success('Tá tranquilo!')
				elif prob < 0.7:
					st.warning('Probabilidade Moderada de Fraude')
				else:
					st.error('Probabilidade Alta de Fraude!')
			




		###### PAGINA: OBSERVAÇÕES ######
		else:
			st.markdown('![alt text](https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide4.JPG?raw=true)')

			st.write("""
					 # Observações
		             """) 

			st.markdown('Nesse Web App mostramos o poder do streamlit para construir \
				soluções fáceis, rápidas e que permitem uma usabilidade bastante ampla.')
				
			st.markdown('Pare por um momento e imagine o universo de possibilidades que temos ao combinar\
				todos os recursos do streamlit com o que já temos no Python.')
				
				
			st.markdown('Esse tipo de web-app é perfeito para quando se quer entregar uma solução rápida\
				e/ou criar um ambiente de testes mais eficiente.')

			st.markdown('Não deixe de explorar os recursos do streamlit. Aprenda, crie, desenvolva. \
				Faça o que ninguém fez ainda. Não se limite, pois...') 

			st.markdown('---')
			
			if st.button('Comemorar'):
				st.balloons()
















########################
######################
#####################
###### PAGINA: OBSERVAÇÕES ######
else:
	st.markdown('![alt text](https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide4.JPG?raw=true)')

	st.write("""
			 # Observações
             """) 

	st.markdown('Nesse Web App mostramos o poder do streamlit para construir \
		soluções fáceis, rápidas e que permitem uma usabilidade bastante ampla.')
		
	st.markdown('Pare por um momento e imagine o universo de possibilidades que temos ao combinar\
		todos os recursos do streamlit com o que já temos no Python.')
		
		
	st.markdown('Esse tipo de web-app é perfeito para quando se quer entregar uma solução rápida\
		e/ou criar um ambiente de testes mais eficiente.')

	st.markdown('Não deixe de explorar os recursos do streamlit. Aprenda, crie, desenvolva. \
		Faça o que ninguém fez ainda. Não se limite, pois...') 

	st.markdown('---')
	
	if st.button('Comemorar'):
		st.balloons()

st.sidebar.markdown('---')  
st.sidebar.markdown(':copyright: Ricardo Rocha') 
st.sidebar.markdown('---')
