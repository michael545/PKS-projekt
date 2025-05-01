# PKS Projektna Naloga: Avtomatizirana Polnilna Linija (TIA Portal)

## Opis Projekta

Ta repozitorij vsebuje projekt za Siemens TIA Portal, razvit kot del projektne naloge pri predmetu PKS (Procesni Krmilni Sistemi). Namen projekta je avtomatizacija in vodenje polnilne linije za pripravo gradbene mase in polnjenje le-te v posode. Projekt vključuje krmilni program za PLC Siemens S7-1200 in uporabniški vmesnik za HMI panel TP700 Comfort.

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

## Avtomat Stanja (Avtomatski Režim)

Logika avtomatskega režima je implementirana kot avtomat stanj v bloku `FB3 rezim_avtomatsko`. Stanja predstavljajo posamezne korake tehnološkega procesa.

```mermaid
graph TD
    subgraph Legenda
        direction LR
        A[Stanje] --> B(Dogodek/Pogoj za prehod)
        C[/Akcija v stanju/]
        D((Končno/Posebno stanje))
    end

    subgraph Glavni Cikel
        IDLE(0: IDLE / PRIPRAVLJEN) -- START & Pripravljen --> SP(10: POSTAVI PALETO)
        SP -- Paleta OK --> SPP(20: POSTAVI POSODO NA P)
        SPP -- Posoda OK & P na S2 --> PP_T(30: PREMAKNI P NA TEHTNICO \(S3\))
        PP_T -- S3 Dosežen --> CALC(40: IZRAČUN KOLIČIN)
        CALC -- Izračunano --> DA_T(50: PREMAKNI DOZ A NAD TEHTNICO \(S10\))
        DA_T -- S10 Dosežen --> FILLA(60: POLNI A)
        FILLA -- Masa A OK --> DA_H(70: PREMAKNI DOZ A DOMOV \(S9\))
        DA_H -- S9 Dosežen --> DB_T(80: PREMAKNI DOZ B NAD TEHTNICO \(S12\))
        DB_T -- S12 Dosežen --> FILLB(90: POLNI B)
        FILLB -- Masa B OK --> DB_H(100: PREMAKNI DOZ B DOMOV \(S11\))
        DB_H -- S11 Dosežen --> M_DOWN(110: PREMAKNI MEŠALO DOL \(S14\))
        M_DOWN -- S14 Dosežen --> MIX(120: MEŠAJ)
        MIX -- Čas Mešanja OK --> M_UP(130: PREMAKNI MEŠALO GOR \(S13\))
        M_UP -- S13 Dosežen --> PP_TR(140: PREMAKNI P NA TRAK \(S4\))
        PP_TR -- S4 Dosežen & Pripravljenost Traku --> P_H_TR(150: PREMAKNI P DOMOV \(S2\) & VKLOP TRAKU)
        P_H_TR -- S2 Dosežen --> TR_S6(160: TRANSPORT DO S6)
        TR_S6 -- S6 Dosežen --> VIB(170: VIBRIRAJ)
        VIB -- S7 Dosežen & Čas Vibracij OK --> TR_S8(180: TRANSPORT DO S8)
        TR_S8 -- S8 Dosežen --> CHK(190: PREVERI ŠTEVILO POSOD)
        CHK -- Števec < 6 --> SPP
        CHK -- Števec = 6 --> FULL((200: PALETA POLNA))
        FULL -- Odstrani Paleto --> IDLE
    end

    subgraph Prekinitve in Napake
        AktivnaStanja -- STOP (Dokončaj) --> CHK_STOP(195: Končaj Posodo Po STOP)
        CHK_STOP --> STOPPED
        AktivnaStanja -- STOP (Takoj) --> STOPPED((998: USTAVLJEN \(Ročni\)))
        AktivnaStanja -- NAPAKA (error_word != 0) --> ERROR((999: NAPAKA))
        VsaStanja -- ZASILNI IZKLOP / Gl. Stikalo OFF --> E_STOPPED((Stanje E-STOP))
        STOPPED -- Reset / Nova Izbira --> IDLE
        ERROR -- Reset --> IDLE
        E_STOPPED -- Sprostitev E-STOP & Gl. Stikalo ON & Reset --> IDLE
    end

    %% Opombe o stilih (ostanejo enake, prilagodite po želji)
    style IDLE fill:#f9f,stroke:#333,stroke-width:2px
    % ... (ostale definicije stilov) ...
    style FULL fill:#f9f,stroke:#333,stroke-width:2px
    style STOPPED fill:#f99,stroke:#333,stroke-width:2px
    style ERROR fill:#f00,stroke:#333,stroke-width:2px
    style E_STOPPED fill:#f60,stroke:#333,stroke-width:2px

    %% Povezave iz skupin stanj (za jasnost sem jih preimenoval)
    linkStyle default interpolate basis
    FILLA --> AktivnaStanja
    FILLB --> AktivnaStanja
    MIX --> AktivnaStanja
    VIB --> AktivnaStanja
    TR_S6 --> AktivnaStanja
    TR_S8 --> AktivnaStanja
    % ... ostala aktivna stanja ... --> AktivnaStanja

    IDLE --> VsaStanja
    SP --> VsaStanja
    % ... vsa stanja ... --> VsaStanja
    FULL --> VsaStanja
    STOPPED --> VsaStanja
    ERROR --> VsaStanja
    E_STOPPED --> VsaStanja

    style AktivnaStanja stroke-width:0px, fill:none, color:none    %% Naredi vozlišče nevidno
    style VsaStanja stroke-width:0px, fill:none, color:none        %% Naredi vozlišče nevidno
```mermaid