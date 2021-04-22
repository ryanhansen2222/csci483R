'use strict'

/*
|--------------------------------------------------------------------------
| Routes
|--------------------------------------------------------------------------
|
| Http routes are entry points to your web application. You can create
| routes for different URL's and bind Controller actions to them.
|
| A complete guide on routing is available here.
| http://adonisjs.com/docs/4.1/routing
|
*/

/** @type {typeof import('@adonisjs/framework/src/Route/Manager')} */
const Route = use('Route')

Route.on('/').render('index')


Route.on('/setup').render('setup.edge')
Route.on('/tf').render('tf.edge')
Route.on('/gl').render('gl.edge')
Route.on('/nextsteps').render('nextsteps.edge')


