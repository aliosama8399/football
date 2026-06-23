import strawberry
from api.graphql.resolvers import Query

schema = strawberry.Schema(query=Query)
