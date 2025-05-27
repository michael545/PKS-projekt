## Super detajl: Določanje karakteristike ventila A za natančen izračun časa doziranja

Pomemben izziv pri tej projektni nalogi je bilo zagotavljanje izjemno natančnega doziranja komponente A (vode). Elektromagnetni ventil A ima spremenljiv kot odprtja (0-90 stopinj) in mrtvi čas pri odpiranju. Zahteva je bila, da se komponenta A dozira "čim hitreje in čim bolj natančno", kar je glede na lastnosti ventila A predstavljalo poseben izziv.

Da bi dosegli visoko natančnost in ponovljivost doziranja komponente A ter hkrati kompenzirali nelinearnosti in mrtvi čas ventila, smo se odločili za podrobno karakterizacijo njegovega delovanja. To je omogočilo razvoj matematičnega modela, ki določi potrebno trajanje odprtosti ventila za doseganje želene mase komponente A (vode) na podlagi željenega volumna posode, mešalnega razmerja ter maksimalnega pretoka ventila A.

### Konceptualni postopek določanja karakteristike in izračuna časa:

1.  **Pridobivanje eksperimentalnih podatkov:**
    * Najprej smo v TIA portalu zbrali podatke o količini (masi) pretočene komponente A in trenutne odprtosti ventila (deg) v odvisnosti od časa (100ms diskretno vzorčenje), ko je bil ventil odprt (doveden HIGH signal na VENTIL_A_ON_OFF).
    * Ti podatki so bili zajeti pri različnih časih odprtosti, da smo dobili celovito sliko dinamičnega odziva ventila. Primer uporabljenih podatkov je v datoteki `ventil_karakteristika.txt`, ki se uporablja v priloženi analitični kodi.

2.  **Modeliranje posameznih faz pretoka:**
    * Na podlagi zbranih podatkov smo analizirali in modelirali različne faze pretoka skozi ventil:
        * **Faza odpiranja in ustaljenega pretoka:** Opisuje, kako se masa naraščene tekočine spreminja od trenutka, ko je ventil dobil ukaz za odprtje, preko mrtvega časa odpiranja, do morebitne ustaljene faze pretoka (ko je dm/dt = const). Za to fazo smo uporabili polinomsko regresijo (spline 2. reda) za opis odvisnosti mase od časa $m_{odpiranja}(t)$. Grafični prikaz te regresije sliki z naslovom, podobnim `Regresija_za_posamezne_faze_-_Dataset_1_(Nominal).png`.
        * **Faza zapiranja (vpliv mrtvega časa):** Opisuje dodatno količino (maso) tekočine, ki še steče skozi ventil potem, ko je bil dan ukaz za zaprtje. Ta "delta masa" ($\Delta m_{zapiranja}$) je ključna za kompenzacijo prekoračitve. Tudi ta del karakteristike smo modelirali, s spline regresijo, ki opisuje to dodatno maso.

3.  **Simulacija skupne pretočene mase:**
    * Z združitvijo modela faze odpiranja in modela delta mase med zapiranjem smo lahko simulirali skupno pretočeno maso za poljubno (ukazano) trajanje odprtosti ventila: $M_{skupna}(t_{ukaz}) = m_{odpiranja}(t_{ukaz}) + \Delta m_{zapiranja}$. Rezultati te simulacije so prikazani na grafu, `Simulirane_komponente_mase_-_Dataset_1_(Nominal).png`.

4.  **Izpeljava inverzne karakteristike (Čas iz Mase):**
    * **Ključni korak** je bil izpeljava inverznega modela, ki nam iz *želene ciljne mase* komponente A ($M_{ciljna}$) izračuna *potrebno trajanje ukaza za odprtje ventila* ($t_{ukaz} = f(M_{ciljna})$).
    * Za to smo uporabili simulirane podatke o skupni masi glede na čas. Ker je ta odvisnost na začetku nelinearna in na drugem odseku linearna, smo za inverzni model uporabili kombinacijo:
        * Polinomske regresije za določen začetni razpon mas (npr. do mase, ki ustreza približno 9 sekundam delovanja ventila, ko se ventil popolnoma odpre).
        * Linearne regresije za mase, ki presegajo ta začetni razpon.
    * Rezultat tega koraka je matematični model (sestavljen iz polinomskih in linearnih koeficientov), ki predstavlja inverzno karakteristiko ventila A. Grafični prikaz te inverzne karakteristike je viden na sliki, `Inverzne_karakteristike-_Dataset_1_(Nominal).png`.

5.  **Implementacija v PLCju:**
    * Pridobljeni koeficienti inverznega modela ($t_{ukaz} = f(M_{ciljna})$) so nato implementirani v logiki krmilnika.
    * Ko avtomatski sistem v skladu s tehnološkim postopkom izračuna zahtevano maso komponente A, na podlagi tega modela natančno določi, kako dolgo mora biti krmilni signal `ventil_A+` aktiven. S tem se učinkovito modelira nelinearnosti ventila, kar vodi do bistveno bolj natančnega doziranja.
