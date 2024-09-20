css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #bdbbb5
}
.chat-message.bot {
    background-color: #b7b0b3
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/jpeg;base64,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wXJh9zpif1c+oDwyP+lZiqfwvKyPiC4Na5uiZjS0/tOGfkrTWTNhpZZHHADT7ey5l7EbZH8+lnlcSeZK52VltuI66pjB2c1rh7wfktCyzsgtccZ18wjLmxxueWk+OAcetZaGqj/S7AeY0yMc1vMjc3J2PceYq7PBE8iIs3RERAREQEREBERAREQEREGrcopJ6KWOGV0UjhhsjerT4qkycH1ragyM4m4gaCckOrC4ewjAV2uEM00GmCQxvyCDqI+CiZKW9kaW1Lselnzbla4a+WOfJcb9tv8ACI4chlgvcrayodKYGZ5rmgGTOwJwMZ3d08FZblNFPQzMZKGv05afAjcKHpeGql9S+praol7sbBxP5KVFnj0EF7unil679rltm9KrPYbldaSOVt7uVFG4ZbDSSCNoHidsknqTnusnD1lls9zhkqrrcquPfIrJjJh3Ygnp3U2233WmZyqepfyx9ny2nb8TThYHWe7VNRG+oqhpY7Jy8fBoAVePllOW2661YW1DHdCsgeD0WpBQ8sDU8krZbHpWVk/DbyyIvAvVKhERAREQEREBERAREQF4iICIiAiIg9REQEREBERAREQf/9k=" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ832M5HMrevQNzF8Yn5VEn6RhghY44kaQF9Q&s">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
