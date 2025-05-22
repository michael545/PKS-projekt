# PKS Projektna Naloga: Avtomatizirana Polnilna Linija (TIA Portal)

## Opis Projekta

Ta repozitorij vsebuje projekt za Siemens TIA Portal, razvit kot del projektne naloge pri predmetu PKS (Progralmirljivi Krmilni Sistemi). Namen projekta je avtomatizacija in vodenje polnilne linije za pripravo gradbene mase in polnjenje le-te v posode. Projekt vključuje krmilni program za PLC Siemens S7-1200 in uporabniški vmesnik za HMI panel TP700 Comfort.

Tehnološki postopek zajema naslednje korake:
1.  Postavitev prazne palete na konec linije.
2.  Postavitev prazne posode na podajalnik P.
3.  Premik posode na polnilno mesto (tehtnica).
4.  Doziranje komponente A (voda) in komponente B (agregat).
5.  Mešanje komponent.
6.  Premik polne posode na tekoči trak.
7.  Transport posode po traku preko vibracijske plošče.
8.  Odlaganje posode na paleto.

## Strojna Oprema

* **PLC:** Siemens SIMATIC S7-1200 (Specifično: S7-1214C DC/DC/DC, IP: 192.168.0.1)
* **HMI:** Siemens SIMATIC TP700 Comfort Panel (IP: 192.168.0.2)

## Programska Oprema

* Siemens TIA Portal

## Struktura Projekta (TIA Portal)

Struktura vključuje naslednje ključne komponente:

* **Organizacijski Bloki (OBs):**
    * `Main [OB1]`: Glavni ciklični program.
    * `Cyclic interrupt [OB30]`: Periodična prekinitev (100 ms) za izvajanje simulacijskega modela in časovno kritičnih nalog.
    * `Startup [OB100]`: Zagon programa in inicializacija spremenljivk.
* **Funkcijski Bloki (FBs):**
    * `FB1 FB_sim_model`: Simulacijski model tehnološkega procesa (zaščiten z geslom).
    * `FB2 IZBERI_REZIM`: Logika za izbiro med ročnim in avtomatskim režimom.
    * `FB3 rezim_avtomatsko`: Implementacija logike za avtomatski režim (vključno z avtomatom stanj).
    * `FB4 rezim_rocno` : Implementacija logike za ročni režim.
    * `FB5 Izracun_Energije`: Izračun porabe energije vibracijske plošče in mešala  [kWH].
* **Funkcije (FCs):**
    * `FC1_init`: Inicializacija (klicano iz OB100).
    * `FC2 resetiraj_aktuatorje`: Funkcija za resetiranje spremenljivk ki aktivirajo aktuatorjev.
    * `FC3 OB30_Network1` / `FC4 OB30_Network4`: prvi network v OB30 , funkcije, klicane iz OB30.
* **Podatkovni Bloki (DBs):**
    * `DB1 FB_sim_model_DB`: Instančni DB za simulacijski model (zaščiten).
    * `DB2 DB_podatki_procesa`: Parametri procesa (npr. `V_žel`, `m_posode`, `mr_AB_žel`).
    * `DB3 DB_spremenljivke`: Globalne spremenljivke za krmiljenje procesa iz HMI HMI (npr. `tipka_start`, `tipka_stop`, stanja).
    * `DB4 DB_izhodne_spremenljivke`: Izhodne spremenljivke iz procesa ki so rezultat simulacijskega teka.
    * `DB5 DB_vhodne_aktuatorjev`: Signali ki krmilijo vhode aktuatorjev iz HMI/logike.
    * `DB6 IZBERI_REZIM_DB`: Instančni DB za FB2.
    * `DB7 Izracun_Energije_DB`: Instančni DB za FB5.
    * `DB8 CHAT_GPT_ROCNO_DB` / `DB10 DB_sim_model` / `DB12 GLOBAL_Sync_Flags`: Dodatni DB-ji, za testiranje.
* **HMI Screens:** Uporabniški vmesnik na TP700 Comfort panelu za spremljanje in upravljanje procesa.

*(Opomba: Struktura temelji na priloženi sliki `image_1a81ac.png` in opisu. Imena nekaterih blokov (npr. `CHAT_GPT...`) morda odražajo razvojno fazo.)*

## Funkcionalnost

Krmilni program omogoča dva načina delovanja:

### 1. Ročni Režim (`rezim_rocno`)

* Privzeti način ob vklopu (če ni napake in ni pritisnjen E-Stop).
* Omogoča neposredno upravljanje posameznih aktuatorjev preko HMI vmesnika:
    * Postavitev/odstranitev palete in posode.
    * Premikanje podajalnikov P in M ter dozirnikov A in B.
    * Vklop/izklop ventilov A in B.
    * Vklop/izklop mešala, tekočega traku in vibracijske plošče.
* Prikaz vseh relevantnih procesnih vrednosti (volumen, masa, stanje senzorjev, napake, poraba energije...).
* Možnost vnosa parametrov (npr. `V_žel`).
* Namenjen testiranju, vzdrževanju in spoznavanju delovanja linije.

### 2. Avtomatski Režim (`rezim_avtomatsko`)

* Aktivira se iz ročnega režima preko HMI tipke, če so izpolnjeni pogoji pripravljenosti (`pripravljenost` = TRUE).
* Po pritisku tipke `START` izvede celoten cikel polnjenja za 6 posod:
    1.  Avtomatska postavitev palete in prve posode.
    2.  Premik posode na tehtnico.
    3.  Izračun in natančno doziranje komponent A in B glede na `V_žel` in `mr_AB_žel`.
    4.  Avtomatsko mešanje (`t_žel_meš`).
    5.  Premik polne posode na trak.
    6.  Avtomatski vklop traku in transport.
    7.  Aktivacija vibracijske plošče za zahtevan čas (`t_žel_vib`), odvisno od mase.
    8.  Odlaganje posode na paleto.
    9.  Ponavljanje korakov 2-8 za naslednjih 5 posod.
* Po napolnitvi 6 posod se cikel ustavi in čaka na odstranitev polne palete (tipka `odstrani_paleto`).
* Možnost prekinitve cikla s tipko `STOP` (proces se zaključi za trenutno posodo) ali takojšnje zaustavitve z `ZASILNI IZKLOP` ali izklopom glavnega stikala.
* Prikaz stanja cikla, števca posod, skupnega časa, odstopanj volumna (`ABSE`), doseženega razmerja (`mr_AB_dej`), prelitega volumna (`V_polit`) in napak.
* Možnost vnosa `V_žel` in `mr_AB_žel`.

## Diagram Stanj (Avtomatski Režim)

Logika avtomatskega režima je implementirana kot sekvenčni stroj v bloku `FB3 rezim_avtomatsko`. Stanja predstavljajo posamezne korake tehnološkega procesa.
```mermaid
graph TD
    subgraph Legenda
        direction LR
        A[Stanje] --> B(Dogodek/Pogoj za prehod)
        C[/Akcija v stanju/]
        D((Končno/Posebno stanje))
    end

    subgraph "Glavni Cikel Avtomatskega Režima (po rezim_avtomatsko.txt)"
        S0["0: IDLE / Pripravljen"] -- "START pritisnjen" --> S10["10: Zahteva Dodaj Paleto"]
        S10 -- "Interni prehod" --> S15["15: Čakaj Potrditev Palete (OB30)"]
        S15 -- "Paleta Potrjena (Ack)" --> S20["20: Zahteva Postavi Posodo"]
        S20 -- "Interni prehod" --> S25["25: Čakaj Potrditev Posode (OB30)"]
        S25 -- "Posoda Potrjena (Ack)" --> S30["30: Premakni P na Tehtnico (do S3)"]
        S30 -- "P na Tehtnici (S3)" --> S40["40: Izračun Količin A in B"]
        S40 -- "Količine Izračunane" --> S45["45: Premakni Doz A na Poln. Mesto (do S10)"]
        S45 -- "Doz A na Poln. Mestu (S10)" --> S50["50: Polni Komponento A (časovno)"]
        S50 -- "Čas Polnjenja A Potekel" --> S55["55: Zapri Ventil A"]
        S55 -- "Ventil A Zaprt (0 deg)" --> S60["60: Premakni Doz A Domov (do S9)"]
        S60 -- "Doz A Doma (S9)" --> S65["65: Premakni Doz B na Poln. Mesto (do S12)"]
        S65 -- "Doz B na Poln. Mestu (S12)" --> S70["70: Polni Komponento B (do ciljne mase)"]
        S70 -- "Ciljna Masa B Dosežena & Ventil B Izklopljen" --> S80["80: Premakni Doz B Domov (do S11)"]
        S80 -- "Doz B Doma (S11)" --> S90["90: Spusti Mešalo (do S14)"]
        S90 -- "Mešalo Spuščeno (S14)" --> S100["100: Mešaj (časovno)"]
        S100 -- "Čas Mešanja Potekel & Motor Mešala Izklopljen" --> S110["110: Dvigni Mešalo (do S13)"]
        S110 -- "Mešalo Dvignjeno (S13)" --> S120["120: Premakni P (s posodo) na Trak (do S4)"]
        S120 -- "P na Začetku Traku (S4)" --> S130["130: Vklop Traku, Premakni P Domov (do S2), Kontrola Vibro (S6-S7)"]
        S130 -- "Posoda na Koncu Traku (S8) & P Doma (S2)" --> S140["140: Odlaganje na Paleto (kratka pavza), Izklop Traku, Povečaj Števec Posod"]
        S140 -- "Končano Odlaganje" --> S150["150: Preveri Ali je Paleta Polna"]
        S150 -- "#Posod < 6" --> S20
        S150 -- "#Posod = 6" --> S155["155: Paleta Polna, Čakaj Zahtevo 'Odstrani Paleto'"]
        S155 -- "Zahteva 'Odstrani Paleto' (Ack iz OB30)" --> S0
    end

    subgraph "Prekinitve in Napake (po rezim_avtomatsko.txt in splošni logiki)"
        VsaAktivnaStanja["Vsa Aktivna Stanja (Stanja S10-S155)"]

        S0 -- "STOP pritisnjen" --> S998
        VsaAktivnaStanja -- "STOP pritisnjen (takojšen prehod v 998, če ni v 'dokončaj sekvenco')" --> S998
        VsaAktivnaStanja -- "STOP pritisnjen (sproži DokončajSekvencoFlag, če 20<=STANJE<=155, normalno nadaljevanje dokler StopTipkaFlag ne povzroči prehoda v S998)" --> VsaAktivnaStanja

        S998["998: USTAVLJEN / Preklop na Ročni Režim"] -- "Interni prehod" --> S0

        SistemskaStanja["Sistemska Stanja (Vsa Stanja Cikla S0-S155, S998)"]
        SistemskaStanja -- "NAPAKA (error_word aktiven)" --> StanjeNapake["Stanje NAPAKE (zahteva Reset)"]
        StanjeNapake -- "Reset" --> S0
        SistemskaStanja -- "ZASILNI IZKLOP / Glavno Stikalo IZKLOP" --> StanjeZasilniIzklop["Stanje ZASILNI IZKLOP (Ročni ob ponovnem zagonu)"]
        StanjeZasilniIzklop -- "Sprostitev & Reset" --> S0
    end

    %% Stili (lahko jih prilagodite)
    style S0 fill:#ccffcc,stroke:#333,stroke-width:2px
    style S155 fill:#ccffcc,stroke:#333,stroke-width:2px
    style S998 fill:#ffcccc,stroke:#333,stroke-width:2px
    style StanjeNapake fill:#ff9999,stroke:#333,stroke-width:2px
    style StanjeZasilniIzklop fill:#ffcc99,stroke:#333,stroke-width:2px
    
    style VsaAktivnaStanja stroke-width:0px, fill:none, color:none
    style SistemskaStanja stroke-width:0px, fill:none, color:none
```

![HMI LAYOUT AVTOMATSKI REZIM](HMI_avtomatsko.png "HMI prototip")
![HMI LAYOUT AVTOMATSKI REZIM](HMI_rocno.png "HMI prototip")