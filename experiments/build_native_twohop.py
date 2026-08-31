#!/usr/bin/env python
"""A two-hop ceiling from facts the model already knows.

The injected composition probe ("the active ingredient of the drug approved
to treat X is") scored 8.2% at 1.7B. That number has no meaning on its own:
if the model composes facts it learned in pretraining at 10%, the injected
knowledge composes as well as native knowledge does, and the finding is about
model size, not about injection. This builds the native reference -- the same
latent two-hop shape, on famous people:

    hop 1   "The country in which Albert Einstein was born is"        -> Germany
    hop 2   "The capital city of Germany is"                           -> Berlin
    two-hop "The capital city of the country in which Albert Einstein
             was born is"                                              -> Berlin

Both hops are probed on the same model, and the ceiling is the two-hop rate
over items where the model got both hops right -- the compositionality gap of
Press et al. (2022), measured here on the exact model being evaluated. The
injected set is filtered the same way by exp_twohop_ceiling.py, so the two
rates are conditioned identically.

Entities are chosen so that the modern country of birth is unambiguous and
its capital is a single city; people born in places whose country has since
changed, or in a country with a disputed capital, are left out. Alternate
spellings and names are accepted (USA, the United States; Kyiv, Kiev).

  python experiments/build_native_twohop.py --out data/native_twohop.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# country -> (accepted country names, accepted capital names)
COUNTRIES = {
    # The capital must carry its D.C.: a model that answers "Seattle,
    # Washington" for where Bill Gates was born would otherwise score a hit
    # on the two-hop question, and at 27B it does answer that way.
    "United States": (["the United States", "United States", "USA", "U.S.",
                       "the US", "America", "the United States of America"],
                      ["Washington, D.C.", "Washington D.C.", "Washington, DC",
                       "Washington DC", "Washington, D. C."]),
    "United Kingdom": (["the United Kingdom", "United Kingdom", "England",
                        "the UK", "UK", "Great Britain", "Britain"],
                       ["London"]),
    "France": (["France"], ["Paris"]),
    "Germany": (["Germany"], ["Berlin"]),
    "Italy": (["Italy"], ["Rome"]),
    "Spain": (["Spain"], ["Madrid"]),
    "Portugal": (["Portugal"], ["Lisbon"]),
    "Argentina": (["Argentina"], ["Buenos Aires"]),
    "Brazil": (["Brazil"], ["Brasília", "Brasilia"]),
    "Russia": (["Russia"], ["Moscow"]),
    "China": (["China", "the People's Republic of China"], ["Beijing"]),
    "Japan": (["Japan"], ["Tokyo"]),
    "India": (["India"], ["New Delhi", "Delhi"]),
    "Canada": (["Canada"], ["Ottawa"]),
    "Australia": (["Australia"], ["Canberra"]),
    "Mexico": (["Mexico"], ["Mexico City"]),
    "South Korea": (["South Korea", "Korea", "the Republic of Korea"],
                    ["Seoul"]),
    "Sweden": (["Sweden"], ["Stockholm"]),
    "Netherlands": (["the Netherlands", "Netherlands", "Holland"],
                    ["Amsterdam"]),
    "Austria": (["Austria"], ["Vienna"]),
    "Switzerland": (["Switzerland"], ["Bern", "Berne"]),
    "Poland": (["Poland"], ["Warsaw"]),
    "Ireland": (["Ireland", "the Republic of Ireland"], ["Dublin"]),
    "Greece": (["Greece"], ["Athens"]),
    "Turkey": (["Turkey", "Türkiye"], ["Ankara"]),
    "Egypt": (["Egypt"], ["Cairo"]),
    "Nigeria": (["Nigeria"], ["Abuja"]),
    "Kenya": (["Kenya"], ["Nairobi"]),
    "Jamaica": (["Jamaica"], ["Kingston"]),
    "Cuba": (["Cuba"], ["Havana"]),
    "Colombia": (["Colombia"], ["Bogotá", "Bogota"]),
    "Chile": (["Chile"], ["Santiago"]),
    "Peru": (["Peru"], ["Lima"]),
    "Norway": (["Norway"], ["Oslo"]),
    "Denmark": (["Denmark"], ["Copenhagen"]),
    "Finland": (["Finland"], ["Helsinki"]),
    "Belgium": (["Belgium"], ["Brussels"]),
    "Czech Republic": (["the Czech Republic", "Czech Republic", "Czechia"],
                       ["Prague"]),
    "Hungary": (["Hungary"], ["Budapest"]),
    "Romania": (["Romania"], ["Bucharest"]),
    "Serbia": (["Serbia"], ["Belgrade"]),
    "Croatia": (["Croatia"], ["Zagreb"]),
    "Ukraine": (["Ukraine"], ["Kyiv", "Kiev"]),
    "New Zealand": (["New Zealand"], ["Wellington"]),
    "Philippines": (["the Philippines", "Philippines"], ["Manila"]),
    "Indonesia": (["Indonesia"], ["Jakarta"]),
    "Pakistan": (["Pakistan"], ["Islamabad"]),
    "Iran": (["Iran", "Persia"], ["Tehran"]),
    "Ethiopia": (["Ethiopia"], ["Addis Ababa"]),
    "Ghana": (["Ghana"], ["Accra"]),
    "Senegal": (["Senegal"], ["Dakar"]),
    "Algeria": (["Algeria"], ["Algiers"]),
    "Lebanon": (["Lebanon"], ["Beirut"]),
    "Iceland": (["Iceland"], ["Reykjavik", "Reykjavík"]),
    "Venezuela": (["Venezuela"], ["Caracas"]),
    "Uruguay": (["Uruguay"], ["Montevideo"]),
    "Singapore": (["Singapore"], ["Singapore"]),
    "Malaysia": (["Malaysia"], ["Kuala Lumpur"]),
    "Nepal": (["Nepal"], ["Kathmandu"]),
    "Vietnam": (["Vietnam"], ["Hanoi"]),
    "Bangladesh": (["Bangladesh"], ["Dhaka"]),
    "Saudi Arabia": (["Saudi Arabia"], ["Riyadh"]),
    "Morocco": (["Morocco"], ["Rabat"]),
}

PEOPLE = {
    "United States": [
        "Barack Obama", "Michael Jackson", "Elvis Presley", "Bill Gates",
        "Steve Jobs", "Taylor Swift", "Oprah Winfrey", "Mark Zuckerberg",
        "Tom Hanks", "Muhammad Ali", "Michael Jordan", "Abraham Lincoln",
        "Thomas Edison", "Ernest Hemingway", "Mark Twain", "Walt Disney",
        "Madonna", "Beyoncé", "Serena Williams", "Neil Armstrong",
        "Martin Luther King Jr.", "Jeff Bezos", "Warren Buffett",
        "Marilyn Monroe", "Frank Sinatra", "Bob Dylan", "Stephen King",
        "Kobe Bryant", "LeBron James", "Tiger Woods"],
    "United Kingdom": [
        "William Shakespeare", "Isaac Newton", "Charles Darwin",
        "Charles Dickens", "Winston Churchill", "Queen Elizabeth II",
        "David Beckham", "Paul McCartney", "John Lennon", "Stephen Hawking",
        "Alan Turing", "J. K. Rowling", "Adele", "Ed Sheeran", "Jane Austen",
        "Alfred Hitchcock", "Daniel Radcliffe", "Harry Kane",
        "Lewis Hamilton", "Mick Jagger"],
    "France": [
        "Napoleon Bonaparte", "Victor Hugo", "Claude Monet", "Louis Pasteur",
        "Coco Chanel", "Zinedine Zidane", "Kylian Mbappé", "Édith Piaf",
        "Jules Verne", "Brigitte Bardot", "Gustave Eiffel",
        "Antoine Griezmann"],
    "Germany": [
        "Albert Einstein", "Ludwig van Beethoven", "Johann Sebastian Bach",
        "Angela Merkel", "Karl Marx", "Johann Wolfgang von Goethe",
        "Michael Schumacher", "Dirk Nowitzki", "Heidi Klum", "Manuel Neuer"],
    "Italy": [
        "Leonardo da Vinci", "Galileo Galilei", "Michelangelo",
        "Christopher Columbus", "Giuseppe Verdi", "Luciano Pavarotti",
        "Sophia Loren", "Andrea Bocelli", "Valentino Rossi",
        "Dante Alighieri"],
    "Spain": [
        "Pablo Picasso", "Rafael Nadal", "Salvador Dalí", "Antoni Gaudí",
        "Penélope Cruz", "Miguel de Cervantes", "Sergio Ramos",
        "Enrique Iglesias"],
    "Portugal": ["Cristiano Ronaldo", "José Mourinho", "Vasco da Gama"],
    "Argentina": ["Lionel Messi", "Diego Maradona", "Pope Francis",
                  "Che Guevara", "Jorge Luis Borges"],
    "Brazil": ["Pelé", "Neymar", "Ayrton Senna", "Gisele Bündchen",
               "Ronaldinho", "Paulo Coelho"],
    "Russia": ["Vladimir Putin", "Leo Tolstoy", "Fyodor Dostoevsky",
               "Yuri Gagarin", "Pyotr Tchaikovsky", "Maria Sharapova",
               "Anton Chekhov"],
    "China": ["Mao Zedong", "Confucius", "Jack Ma", "Yao Ming", "Xi Jinping",
              "Lang Lang", "Liu Xiang"],
    "Japan": ["Hayao Miyazaki", "Shinzo Abe", "Akira Kurosawa",
              "Shohei Ohtani", "Haruki Murakami", "Naomi Osaka", "Yoko Ono",
              "Ichiro Suzuki"],
    "India": ["Mahatma Gandhi", "Sachin Tendulkar", "Narendra Modi",
              "Rabindranath Tagore", "Virat Kohli", "Amitabh Bachchan",
              "Shah Rukh Khan", "Priyanka Chopra", "Satya Nadella",
              "Sundar Pichai"],
    "Canada": ["Justin Bieber", "Celine Dion", "Wayne Gretzky",
               "Ryan Reynolds", "Jim Carrey", "Drake", "Justin Trudeau",
               "Shania Twain"],
    "Australia": ["Hugh Jackman", "Chris Hemsworth", "Steve Irwin",
                  "Kylie Minogue", "Cate Blanchett", "Rupert Murdoch",
                  "Margot Robbie"],
    "Mexico": ["Frida Kahlo", "Salma Hayek", "Guillermo del Toro",
               "Diego Rivera", "Carlos Slim"],
    "South Korea": ["Son Heung-min", "Ban Ki-moon", "Bong Joon-ho",
                    "Kim Yuna"],
    "Sweden": ["Zlatan Ibrahimović", "Alfred Nobel", "Greta Thunberg",
               "Ingmar Bergman", "Björn Borg"],
    "Netherlands": ["Vincent van Gogh", "Rembrandt", "Johan Cruyff",
                    "Arjen Robben"],
    "Austria": ["Wolfgang Amadeus Mozart", "Arnold Schwarzenegger",
                "Niki Lauda", "Christoph Waltz"],
    "Switzerland": ["Roger Federer", "Carl Jung", "Jean-Jacques Rousseau"],
    "Poland": ["Frédéric Chopin", "Marie Curie", "Nicolaus Copernicus",
               "Robert Lewandowski", "Pope John Paul II"],
    "Ireland": ["Oscar Wilde", "Bono", "James Joyce", "Conor McGregor",
                "Bram Stoker"],
    "Greece": ["Aristotle", "Plato", "Socrates", "Giannis Antetokounmpo"],
    "Turkey": ["Orhan Pamuk"],
    "Egypt": ["Mohamed Salah", "Omar Sharif", "Naguib Mahfouz",
              "Gamal Abdel Nasser"],
    "Nigeria": ["Chinua Achebe", "Wole Soyinka", "Burna Boy"],
    "Kenya": ["Eliud Kipchoge", "Wangari Maathai"],
    "Jamaica": ["Usain Bolt", "Bob Marley"],
    "Cuba": ["Fidel Castro"],
    "Colombia": ["Shakira", "Gabriel García Márquez", "James Rodríguez"],
    "Chile": ["Pablo Neruda", "Gabriela Mistral"],
    "Peru": ["Mario Vargas Llosa"],
    "Norway": ["Edvard Munch", "Henrik Ibsen", "Magnus Carlsen"],
    "Denmark": ["Hans Christian Andersen", "Niels Bohr",
                "Søren Kierkegaard"],
    "Finland": ["Kimi Räikkönen", "Jean Sibelius", "Linus Torvalds"],
    "Belgium": ["Eden Hazard", "Kevin De Bruyne", "Audrey Hepburn",
                "Jean-Claude Van Damme", "Hergé"],
    "Czech Republic": ["Franz Kafka", "Antonín Dvořák", "Milan Kundera",
                       "Petr Čech"],
    "Hungary": ["Ferenc Puskás", "Ernő Rubik"],
    "Romania": ["Nadia Comăneci", "Simona Halep"],
    "Serbia": ["Novak Djokovic", "Nikola Jokić"],
    "Croatia": ["Luka Modrić"],
    "Ukraine": ["Andriy Shevchenko", "Oleksandr Usyk"],
    "New Zealand": ["Edmund Hillary", "Peter Jackson", "Lorde"],
    "Philippines": ["Manny Pacquiao", "Lea Salonga"],
    "Indonesia": ["Joko Widodo", "Sukarno"],
    "Pakistan": ["Malala Yousafzai", "Imran Khan", "Benazir Bhutto"],
    "Iran": ["Asghar Farhadi", "Omar Khayyam"],
    "Ethiopia": ["Haile Gebrselassie", "Abebe Bikila"],
    "Ghana": ["Kofi Annan"],
    "Senegal": ["Sadio Mané"],
    "Algeria": ["Albert Camus"],
    "Lebanon": ["Khalil Gibran"],
    "Iceland": ["Björk"],
    "Venezuela": ["Simón Bolívar"],
    "Uruguay": ["Luis Suárez"],
    "Singapore": ["Lee Kuan Yew"],
    "Malaysia": ["Michelle Yeoh"],
    "Nepal": ["Gautama Buddha"],
    "Vietnam": ["Ho Chi Minh"],
    "Bangladesh": ["Muhammad Yunus"],
    "Saudi Arabia": ["Mohammed bin Salman"],
    "Morocco": ["Ibn Battuta"],
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/native_twohop.json")
    return p.parse_args()


def main():
    args = parse()
    items = []
    for country, people in PEOPLE.items():
        names, capitals = COUNTRIES[country]
        cname = names[0]          # "the United States", not "United States"
        for person in people:
            # Two phrasings of every probe. The continuation form is what
            # the injected probes use; on a base model the same form reads
            # as a quiz stem ("... was born is ____ A. B. C.") and scored
            # 0/264 on hop 1 at 1.7B. The question form avoids that.
            items.append(dict(
                person=person, country=country,
                hop1_prompt=f"The country in which {person} was born is",
                hop1_prompt_qa=f"Q: In which country was {person} born?\nA:",
                hop1_answers=names,
                hop2_prompt=f"The capital city of {cname} is",
                hop2_prompt_qa=f"Q: What is the capital city of {cname}?\nA:",
                hop2_answers=capitals,
                twohop_prompt=(f"The capital city of the country in which "
                               f"{person} was born is"),
                twohop_prompt_qa=(f"Q: What is the capital city of the "
                                  f"country in which {person} was born?\nA:"),
                twohop_answers=capitals))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"{len(items)} people across {len(PEOPLE)} countries -> {args.out}")


if __name__ == "__main__":
    main()
