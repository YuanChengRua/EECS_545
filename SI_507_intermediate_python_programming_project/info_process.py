# First data source
'''
YELP Fusion: We can sequentially ask users questions which will lead to a specific URL for yelp
several questions could include
1. Location
2. Price
3. type of food
4. distance
'''
#
import requests
from flask import Flask, render_template, request, Blueprint, redirect, url_for, session
import json
import sqlite3
import re
from bs4 import BeautifulSoup
import credentials
import uuid
from geopy.geocoders import Nominatim
import random
conn = sqlite3.connect('user_temp.db', check_same_thread=False)
cur = conn.cursor()



app_data = Blueprint('check_point_file', __name__)

@app_data.route('/')
def home():
    return render_template('home_page.html')


@app_data.route('/form_category_and_user_location')
def form_user_category():
    return render_template('auto-complete.html') # just the static HTML

@app_data.route('/handle_form_category_and_user_location', methods=['POST'])
def generate_url():
    # user_pickup_or_delivery = request.form["options"]
    graph_or_not = request.form['graph']

    user_category_list = request.form.getlist('option')
    user_location = request.form["user_address"]
    loc = Nominatim(user_agent="GetLoc")
    getLoc = loc.geocode(user_location)
    session['lat'] = getLoc.latitude
    session['lon'] = getLoc.longitude


    user_phone = request.form['phone']
    session['location'] = user_location
    session['phone'] = user_phone
    user_location_val_temp = user_location.split(',')[0].split()
    user_location_val = "_".join(user_location_val_temp)

    user_category = ",".join(user_category_list)
    #
    baseurl = "https://api.yelp.com/v3/businesses/search?categories=" + user_category + "&location=" + user_location + "&limit=50&sort_by=distance"
    response = requests.get(baseurl, headers=credentials.headers)
    all_info = json.loads(response.text)['businesses']


    field_names = ('name', 'rating', 'distance', 'price', 'transactions', 'image_url', 'categories', 'url', 'display_location', 'phone')
    table_name = 'user_table' + user_location_val
    session['table_name'] = table_name
    conn.execute('DROP TABLE IF EXISTS ' + table_name)
    conn.execute(
        'CREATE TABLE ' + table_name + ' (name_id integer primary key AUTOINCREMENT, name TEXT, rating REAL, distance REAL, price REAL, transactions TEXT, image_url TEXT, categories TEXT, url TEXT, display_location TEXT, phone TEXT)')

    session['lat_list'] = []
    session['lon_list'] = []
    session['res_name_list'] = []
    session['distance'] = []
    for ele in all_info:
        if 'price' not in ele:
            ele['price'] = ''
        insert_data = (
        ele['name'], str(ele['rating']), str(ele['distance']), str(ele['price']), ",".join(ele['transactions']),
        ele['image_url'], ",".join(e['alias'] for e in ele['categories']), ele['url'], ",".join(ele['location']['display_address']), ele['phone'])
        conn.execute('INSERT INTO ' + table_name + '(' + ','.join(field_names) + ') VALUES (' + ','.join('?' * len(insert_data)) + ')',
            insert_data)
        session['lat_list'].append(ele['coordinates']['latitude'])
        session['lon_list'].append(ele['coordinates']['longitude'])
        session['res_name_list'].append(ele['name'])
        session['distance'].append(int(ele['distance']))

    session['total_list'] = []
    for i in range(len(session['res_name_list'])):
        session['total_list'].append([session['res_name_list'][i], session['lat_list'][i], session['lon_list'][i], i+1])

    session['res_dist_list'] = list(zip(session['res_name_list'], session['distance']))
    conn.commit()

    if graph_or_not == 'yes':
        return redirect(url_for('check_point_file.graph_show'))
    else:
        return redirect(url_for('check_point_file.info_show'))

@app_data.route('/info_show')
def info_show():
    return render_template('info_show.html', res_info = session['res_dist_list'])

@app_data.route('/graph_show')
def graph_show():
    return render_template('generate_map.html', user_lat=session['lat'], user_lon=session['lon'], total_list=session['total_list'])

def isLeaf(tree):
    parent = tree[0]
    left_child = tree[1]
    right_child = tree[2]
    if left_child is None and right_child is None:
        return True
    else:
        return False

def playLeaf(tree):
    parent = tree[0]
    left_child = tree[1]
    right_child = tree[2]
    return parent

def simplePlay(tree):
    parent = tree[0]
    left_child = tree[1]
    right_child = tree[2]
    if isLeaf(tree) is False:
        print(parent)
        ans = input("Your answer: ")
        if ans.lower() == "yes":
            return simplePlay(left_child)
        if ans.lower() == "no":
            return simplePlay(right_child)
    else:
        return playLeaf(tree)






@app_data.route('/process_data')
def process_data():
    flag = True
    conn1 = sqlite3.connect('user_temp.db', check_same_thread=False)
    cur = conn1.cursor()
    temp = session['table_name']
    name = cur.execute('SELECT name FROM {} LIMIT 1'.format(temp)).fetchall()[0][0]
    image_url = cur.execute('SELECT image_url FROM {} ORDER BY distance LIMIT 1 '.format(temp)).fetchall()[0][0]
    display_location = \
    cur.execute('SELECT display_location FROM {} ORDER BY distance LIMIT 1 '.format(temp)).fetchall()[0][0]
    phone = cur.execute('SELECT phone FROM {} ORDER BY distance LIMIT 1 '.format(temp)).fetchall()[0][0]

    try:
        while flag:
            mediumTree = \
                ("Do you want to select a restaurant here?",
                 ("Is it " + name + "?",
                  ('I got it', None, None),
                  ('Need to go a little further?',
                   ('Adjusting distance for you', None, None),
                   ('Need to get a higher rating restaurant?', ('Adjusting rating for you', None, None),
                    ('Want to find a cheaper restaurant?', ('Adjusting price for you', None, None),
                     ('Do you want to deliver the food for you?', ('selecting delivery for you', None, None), ('sele cting pickup for you', None, None)))))),
                 ('Please select preference again', None, None))
            jsonObj = json.dumps(mediumTree)
            with open('json_tree.json', 'w') as outfile:
                outfile.write(jsonObj)

            tree = json.loads(jsonObj)

            output = simplePlay(tree)
            if output == 'Please select preference again':
                return redirect(url_for('/form_category_and_user_location'))
            if output == 'I got it':
                t = session['table_name']

                session['name'] = name
                session['image_url'] = image_url
                session['display_location'] = display_location
                session['phone'] = phone
                return redirect(url_for('check_point_file.generate_menu'))
            elif output.split()[1] == 'distance':
                t = session['table_name']
                cur.execute('DELETE FROM {} WHERE name = ?'.format(t), (name,))
                name = cur.execute('SELECT name FROM {} LIMIT 1'.format(t)).fetchall()[0][0]
                image_url = \
                cur.execute('SELECT image_url FROM {} LIMIT 1 '.format(t)).fetchall()[0][0]
                display_location = \
                    cur.execute('SELECT display_location FROM {} LIMIT 1 '.format(t)).fetchall()[0][0]
                phone = cur.execute('SELECT phone FROM {} LIMIT 1 '.format(t)).fetchall()[0][0]

                conn1.commit()

            elif output.split()[1] == 'rating':
                t = session['table_name']
                cur.execute('DELETE FROM {} WHERE name =?'.format(t), (name,))
                query_name = '''
                            SELECT name
                            FROM {}
                            ORDER BY rating DESC, distance
                            LIMIT 1
                        '''
                query_image_url = '''
                            SELECT image_url
                            FROM {}
                            ORDER BY rating DESC, distance
                            LIMIT 1
                        '''
                query_display_location = '''
                            SELECT display_location
                            FROM {}
                            ORDER BY rating DESC, distance
                            LIMIT 1
                        '''
                query_phone = '''
                            SELECT phone
                            FROM {}
                            ORDER BY rating DESC, distance
                            LIMIT 1
                '''
                name = cur.execute(query_name.format(t)).fetchall()[0][0]
                image_url = cur.execute(query_image_url.format(t)).fetchall()[0][0]
                display_location = cur.execute(query_display_location.format(t)).fetchall()[0][0]
                phone = cur.execute(query_phone.format(t)).fetchall()[0][0]
                conn1.commit()
            elif output.split()[1] == 'price':
                t = session['table_name']
                cur.execute('DELETE FROM {} WHERE name =?'.format(t), (name,))
                query_name = '''
                            SELECT name
                            FROM {}
                            ORDER BY price DESC, distance
                            LIMIT 1
                '''
                query_image_url = '''
                            SELECT image_url
                            FROM {}
                            ORDER BY price DESC, distance
                            LIMIT 1
                '''
                query_display_location = '''
                            SELECT display_location
                            FROM {}
                            ORDER BY price DESC, distance
                            LIMIT 1
                '''
                query_phone = '''
                            SELECT phone
                            FROM {}
                            ORDER BY price DESC, distance
                            LIMIT 1
                '''
                name = cur.execute(query_name.format(t)).fetchall()[0][0]
                image_url = cur.execute(query_image_url.format(t)).fetchall()[0][0]
                display_location = cur.execute(query_display_location.format(t)).fetchall()[0][0]
                phone = cur.execute(query_phone.format(t)).fetchall()[0][0]
                conn1.commit()

            elif output.split()[1] == 'delivery':
                print(output)
                t = session['table_name']
                cur.execute('DELETE FROM {} WHERE name =?'.format(t), (name,))
                query_name = '''
                             SELECT name
                             FROM {}
                             WHERE transactions = 'delivery' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                query_image_url = '''
                             SELECT image_url
                             FROM {}
                             WHERE transactions = 'delivery' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                query_display_location = '''
                             SELECT display_location
                             FROM {}
                             WHERE transactions = 'delivery' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                query_phone = '''
                             SELECT phone
                             FROM {}
                             WHERE transactions = 'delivery' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                name = cur.execute(query_name.format(t)).fetchall()[0][0]
                image_url = cur.execute(query_image_url.format(t)).fetchall()[0][0]
                display_location = cur.execute(query_display_location.format(t)).fetchall()[0][0]
                phone = cur.execute(query_phone.format(t)).fetchall()[0][0]

                conn1.commit()

            elif output.split()[1] == 'pickup':
                print(output)
                t = session['table_name']
                cur.execute('DELETE FROM {} WHERE name =?'.format(t), (name,))
                query_name = '''
                             SELECT name
                             FROM {}
                             WHERE transactions = '' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                query_image_url = '''
                             SELECT image_url
                             FROM {}
                             WHERE transactions = '' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                query_display_location = '''
                             SELECT display_location
                             FROM {}
                             WHERE transactions = '' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                query_phone = '''
                             SELECT phone
                             FROM {}
                             WHERE transactions = '' or transactions = 'delivery,pickup'
                             LIMIT 1
                 '''
                name = cur.execute(query_name.format(t)).fetchall()[0][0]
                image_url = cur.execute(query_image_url.format(t)).fetchall()[0][0]
                display_location = cur.execute(query_display_location.format(t)).fetchall()[0][0]
                phone = cur.execute(query_phone.format(t)).fetchall()[0][0]
                conn1.commit()

                # print(cur.execute('SELECT * FROM user_table1697_Broadway_Street').fetchall())
    except:
        return redirect(url_for('check_point_file.form_user_category'))

@app_data.route('/generate_menu')
def generate_menu():
    image = session['image_url']
    name = session['name'].strip().lower()
    location = session['location'].strip().lower()
    city = location.split(',')[1].strip()
    complete = name + '-' + city
    complete = re.sub(" ", "-", complete)
    url = "https://www.yelp.com/menu/" + complete
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    all_dish = soup.find_all(lambda tag: tag.name == 'h4')

    dish_name_list = [ele.text.strip() for ele in all_dish]
    if len(dish_name_list) == 0:
        print(session['name'])
        return render_template('error_mess.html', res_name = session['name'])

    else:
        return render_template('generate_menu_form.html', dish_name_list=dish_name_list, image_url=image, res_name = session['name'])

@app_data.route('/handle_form_menu', methods=['POST'])
def user_menu():
    # user_pickup_or_delivery = request.form["options"]
    dish_list = request.form.getlist('dish')
    session['dish_list'] = dish_list

    return redirect(url_for('check_point_file.start_delivery'))


@app_data.route('/start_delivery')
def start_delivery():
    token = credentials.token
    res_location = session['display_location']
    user_location = session['location']
    endpoint = "https://openapi.doordash.com/drive/v2/deliveries" # DRIVE API V2
    headers = {"Authorization": "Bearer " + token,
                "Content-Type": "application/json"}

    delivery_id = str(uuid.uuid4()) # Randomly generated UUID4  # Note, this is a delivery simulation

    request_body = {
        "external_delivery_id": delivery_id,
        "pickup_address": res_location,
        "pickup_business_name": session['name'],
        "pickup_phone_number": session['phone'],
        "dropoff_address": session['location'],
        "dropoff_business_name": "User",
        "dropoff_phone_number": session['phone'],
        "order_value":random.randint(1000, 10000)

    }
    create_delivery = requests.post(endpoint, headers=headers, json=request_body) # Create POST request

    return redirect(json.loads(create_delivery.text)['tracking_url'])


















    # return render_template('generate_map.html', user_loc = user_location, user_pickup_or_delivery=user_pickup_or_delivery,
    #                        min_price = user_min_price, max_price = user_max_price, category = user_category)



#
if __name__ == '__main__':

    app_data.run(debug=True)
